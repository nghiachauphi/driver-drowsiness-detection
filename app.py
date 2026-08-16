"""Realtime driver-drowsiness detection with the E6 MobileNetV2 model."""
import csv
import io
from pathlib import Path
from functools import lru_cache
import threading
import tempfile
import zipfile

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "models" / "E6_MobileNetV2_subject.keras"
INPUT_SIZE = 96


@lru_cache(maxsize=8)
def load_unicode_font(size):
    """Load a Vietnamese-capable font on Windows, Linux, or macOS."""
    font_candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)

    # Pillow normally bundles DejaVu Sans, so this also works on many minimal images.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_unicode_text(image_bgr, text, position, color_bgr, font_size=24):
    """Draw UTF-8 text on a BGR OpenCV frame and return the updated frame."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    color_rgb = tuple(reversed(color_bgr))
    draw.text(
        position,
        text,
        font=load_unicode_font(font_size),
        fill=color_rgb,
        stroke_width=1,
        stroke_fill=color_rgb,
    )
    return cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)

st.set_page_config(
    page_title="Phân tích trạng thái buồn ngủ",
    page_icon=":material/visibility:",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp mô hình MobileNetV2…")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH}")
    # The notebook saved MobileNetV2 preprocessing inside a Lambda layer.
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
    )


@st.cache_resource
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self, model, detector, threshold):
        self.model = model
        self.detector = detector
        self.threshold = threshold
        self.lock = threading.Lock()
        self.last_probability = None
        self.last_face_count = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        probability, face_count, face_box = predict_frame(
            image, self.model, self.detector
        )
        image = draw_prediction(image, probability, face_box, self.threshold)

        with self.lock:
            self.last_probability = probability
            self.last_face_count = face_count
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def predict_frame(image_bgr, model, detector):
    """Return probability, face count, and the largest detected face box."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if not len(faces):
        return None, 0, None

    # Prefer the largest face when several people are in view.
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face_rgb = cv2.cvtColor(image_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
    probability = float(model(batch, training=False).numpy()[0, 0])
    return probability, len(faces), (x, y, w, h)


def draw_prediction(image_bgr, probability, face_box, threshold):
    """Draw one prediction using a Vietnamese-capable Unicode font."""
    result = image_bgr.copy()
    if face_box is None:
        return draw_unicode_text(
            result,
            "Không phát hiện khuôn mặt",
            (20, 12),
            (0, 165, 255),
        )

    x, y, w, h = face_box
    drowsy = probability >= threshold
    label = "CẢNH BÁO: BUỒN NGỦ" if drowsy else "TỈNH TÁO"
    color = (0, 0, 255) if drowsy else (0, 190, 0)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
    return draw_unicode_text(
        result,
        f"{label}: {probability:.1%}",
        (x, max(4, y - 34)),
        color,
    )


def annotate_frame(image_bgr, model, detector, threshold):
    """Detect the largest face and draw its prediction on one BGR frame."""
    probability, face_count, face_box = predict_frame(image_bgr, model, detector)
    result = draw_prediction(image_bgr, probability, face_box, threshold)
    return result, probability, face_count


def analyze_uploaded_video(
    video_bytes,
    suffix,
    model,
    detector,
    threshold,
    analysis_rate,
    alert_interval,
    max_alert_images,
    progress_callback=None,
):
    """Analyze sampled frames and extract threshold-crossing snapshots."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source_file:
        source_file.write(video_bytes)
        source_path = source_file.name
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    capture = None
    output = None
    try:
        capture = cv2.VideoCapture(source_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("Không đọc được dữ liệu video.")

        output = av.open(output_path, mode="w", format="mp4")
        video_stream = output.add_stream("libx264", rate=max(1, round(fps)))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        sample_stride = max(1, round(fps / analysis_rate))
        alert_snapshots = []
        probability_points = []
        total_frames = analyzed_frames = face_frames = alert_frames = 0
        last_snapshot_time = -alert_interval
        current_probability = None
        current_face_box = None
        progress_step = max(1, expected_frames // 100) if expected_frames > 0 else 30

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number = total_frames
            total_frames += 1

            if frame_number % sample_stride == 0:
                analyzed_frames += 1
                timestamp = frame_number / fps
                current_probability, face_count, current_face_box = predict_frame(
                    frame, model, detector
                )
                if face_count:
                    face_frames += 1
                    probability_points.append((timestamp, current_probability))
                    if current_probability >= threshold:
                        alert_frames += 1
                        can_save = timestamp - last_snapshot_time >= alert_interval
                        if can_save and len(alert_snapshots) < max_alert_images:
                            annotated = draw_prediction(
                                frame,
                                current_probability,
                                current_face_box,
                                threshold,
                            )
                            encoded, jpeg = cv2.imencode(
                                ".jpg",
                                annotated,
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )
                            if encoded:
                                alert_snapshots.append(
                                    {
                                        "frame_number": frame_number,
                                        "time_seconds": timestamp,
                                        "probability": current_probability,
                                        "image": jpeg.tobytes(),
                                    }
                                )
                                last_snapshot_time = timestamp

            annotated_frame = draw_prediction(
                frame,
                current_probability,
                current_face_box,
                threshold,
            )
            video_frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            for packet in video_stream.encode(video_frame):
                output.mux(packet)

            if progress_callback and total_frames % progress_step == 0:
                progress_callback(total_frames, expected_frames)

        if total_frames == 0:
            raise ValueError("Video không có khung hình để xử lý.")
        for packet in video_stream.encode():
            output.mux(packet)
        output.close()
        output = None
        processed_video = Path(output_path).read_bytes()
        if progress_callback:
            progress_callback(total_frames, total_frames)
        return {
            "processed_video": processed_video,
            "fps": fps,
            "duration_seconds": total_frames / fps,
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "face_frames": face_frames,
            "alert_frames": alert_frames,
            "probability_points": probability_points,
            "alert_snapshots": alert_snapshots,
        }
    finally:
        if capture is not None:
            capture.release()
        if output is not None:
            output.close()
        Path(source_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def build_alert_archive(alert_snapshots, threshold):
    """Create a ZIP containing alert JPEGs and a UTF-8 CSV manifest."""
    archive_buffer = io.BytesIO()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["STT", "Tệp ảnh", "Khung hình", "Thời điểm (giây)", "Xác suất", "Ngưỡng"])

    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, alert in enumerate(alert_snapshots, start=1):
            filename = (
                f"canh_bao_{index:03d}_t{alert['time_seconds']:09.2f}s_"
                f"p{alert['probability'] * 100:05.1f}.jpg"
            )
            archive.writestr(f"anh_canh_bao/{filename}", alert["image"])
            writer.writerow(
                [
                    index,
                    filename,
                    alert["frame_number"],
                    f"{alert['time_seconds']:.2f}",
                    f"{alert['probability']:.6f}",
                    f"{threshold:.6f}",
                ]
            )
        archive.writestr("bao_cao_canh_bao.csv", "\ufeff" + csv_buffer.getvalue())
    return archive_buffer.getvalue()


def format_video_time(seconds):
    """Format seconds as MM:SS.s or HH:MM:SS.s."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes):02d}:{secs:04.1f}"


st.title("Phân tích trạng thái buồn ngủ của tài xế")
st.caption(
    "Nhận diện khuôn mặt bằng Haar Cascade và phân loại bằng MobileNetV2 (E6)"
)
st.markdown(
    ":blue-badge[MobileNetV2 · E6] "
    ":gray-badge[Đầu vào 96 × 96 px] "
    ":orange-badge[Mô hình thử nghiệm]"
)

with st.container(border=True):
    mode = st.segmented_control(
        "Nguồn dữ liệu",
        ["Tải ảnh/video", "Webcam"],
        default="Tải ảnh/video",
        selection_mode="single",
        required=True,
    )

with st.sidebar:
    st.markdown("### :material/tune: Cấu hình phân tích")
    threshold = st.slider(
        "Ngưỡng cảnh báo",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f",
        help="Mẫu có xác suất bằng hoặc lớn hơn ngưỡng sẽ được đánh dấu cảnh báo.",
    )
    st.metric("Ngưỡng đang dùng", f"{threshold:.0%}")

    if mode == "Tải ảnh/video":
        st.divider()
        st.markdown("#### :material/movie: Lấy mẫu video")
        analysis_rate = st.slider(
            "Tần suất phân tích (mẫu/giây)",
            min_value=1,
            max_value=15,
            value=5,
            help="Tần suất cao cho kết quả dày hơn nhưng xử lý lâu hơn.",
        )
        alert_interval = st.slider(
            "Khoảng cách giữa hai ảnh cảnh báo (giây)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Giảm số ảnh gần giống nhau trong một giai đoạn buồn ngủ liên tục.",
        )
        max_alert_images = st.slider(
            "Số ảnh cảnh báo tối đa",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )
    else:
        analysis_rate, alert_interval, max_alert_images = 5, 1.0, 50

    st.divider()
    st.warning(
        "Mô hình đang ở giai đoạn thử nghiệm. Kết quả chỉ phục vụ nghiên cứu, "
        "không thay thế thiết bị an toàn trên xe."
    )

try:
    model = load_model()
    detector = load_detector()
except Exception as exc:
    st.error(f"Không thể nạp mô hình: {exc}", icon=":material/error:")
    st.stop()

st.session_state.setdefault("video_analysis", None)
st.session_state.setdefault("video_analysis_key", None)

if mode == "Webcam":
    with st.container(border=True):
        st.markdown("### :material/videocam: Phân tích trực tiếp")
        st.info(
            "Bật webcam, cấp quyền camera cho trang `localhost`, sau đó nhấn **START**. "
            "Ảnh cảnh báo tách riêng hiện áp dụng cho video tải lên."
        )
        camera_enabled = st.toggle(
            "Khởi tạo webcam",
            value=False,
            help="Chỉ bật sau khi webcam đã được kết nối và được Windows nhận.",
        )
        if camera_enabled:
            ctx = webrtc_streamer(
                key="drowsiness-camera",
                video_processor_factory=lambda: DrowsinessProcessor(
                    model, detector, threshold
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    probability = ctx.video_processor.last_probability
                    face_count = ctx.video_processor.last_face_count
                metric_columns = st.columns(3)
                metric_columns[0].metric("Khuôn mặt", face_count)
                metric_columns[1].metric(
                    "Xác suất buồn ngủ",
                    "—" if probability is None else f"{probability:.1%}",
                )
                metric_columns[2].metric("Ngưỡng cảnh báo", f"{threshold:.0%}")
        else:
            st.caption(
                "Nếu trình phát báo `NotFoundError: Requested device not found`, "
                "hãy kiểm tra camera trong Windows hoặc chuyển sang tải ảnh/video."
            )
else:
    with st.container(border=True):
        st.markdown("### :material/upload_file: Dữ liệu đầu vào")
        st.write(
            "Tải ảnh để nhận dạng một thời điểm, hoặc tải video để tách riêng "
            "các ảnh vượt ngưỡng cảnh báo."
        )
        upload = st.file_uploader(
            "Chọn ảnh hoặc video",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
            max_upload_size=200,
            key="media_upload",
        )

    if upload is not None:
        upload_mime = upload.type or ""
        is_image = upload_mime.startswith("image/") or upload.name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
        if is_image:
            raw = np.frombuffer(upload.getvalue(), np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Không thể đọc ảnh đã tải lên.", icon=":material/error:")
            else:
                annotated, probability, face_count = annotate_frame(
                    image, model, detector, threshold
                )
                with st.container(border=True):
                    st.markdown("### :material/image_search: Kết quả phân tích ảnh")
                    if probability is None:
                        st.warning("Không phát hiện khuôn mặt trong ảnh.")
                    elif probability >= threshold:
                        st.error(
                            f"Cảnh báo buồn ngủ — xác suất {probability:.1%} "
                            f"vượt ngưỡng {threshold:.0%}."
                        )
                    else:
                        st.success(
                            f"Chưa vượt ngưỡng cảnh báo — xác suất {probability:.1%}."
                        )

                    metric_columns = st.columns(3)
                    metric_columns[0].metric("Khuôn mặt", face_count)
                    metric_columns[1].metric(
                        "Xác suất buồn ngủ",
                        "—" if probability is None else f"{probability:.1%}",
                    )
                    metric_columns[2].metric("Ngưỡng cảnh báo", f"{threshold:.0%}")

                    image_columns = st.columns(2)
                    image_columns[0].image(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc"
                    )
                    image_columns[1].image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Kết quả nhận dạng",
                    )
        else:
            video_bytes = upload.getvalue()
            analysis_key = (
                upload.name,
                upload.size,
                threshold,
                analysis_rate,
                alert_interval,
                max_alert_images,
            )
            if st.session_state.video_analysis_key != analysis_key:
                st.session_state.video_analysis = None
                st.session_state.video_analysis_key = None

            with st.container(border=True):
                st.markdown("### :material/tune: Thiết lập xử lý video")
                st.caption(
                    f"Phân tích {analysis_rate} mẫu/giây · lưu ảnh cách nhau tối thiểu "
                    f"{alert_interval:g} giây · tối đa {max_alert_images} ảnh"
                )
                analyze_clicked = st.button(
                    "Phân tích video và tách ảnh cảnh báo",
                    type="primary",
                    icon=":material/play_arrow:",
                )

            if analyze_clicked:
                progress = st.progress(0, text="Đang chuẩn bị video…")

                def update_progress(done, total):
                    ratio = min(done / total, 1.0) if total > 0 else 0.0
                    progress.progress(
                        ratio,
                        text=f"Đang phân tích khung hình {done:,}/{total:,}…"
                        if total > 0
                        else f"Đã đọc {done:,} khung hình…",
                    )

                try:
                    st.session_state.video_analysis = analyze_uploaded_video(
                        video_bytes,
                        Path(upload.name).suffix,
                        model,
                        detector,
                        threshold,
                        analysis_rate,
                        alert_interval,
                        max_alert_images,
                        update_progress,
                    )
                    st.session_state.video_analysis_key = analysis_key
                except Exception as exc:
                    st.session_state.video_analysis = None
                    st.session_state.video_analysis_key = None
                    st.error(f"Không thể xử lý video: {exc}", icon=":material/error:")
                finally:
                    progress.empty()

            analysis = st.session_state.video_analysis
            analysis_is_current = (
                analysis is not None
                and st.session_state.video_analysis_key == analysis_key
            )

            with st.container(border=True):
                st.markdown("### :material/compare: So sánh video")
                video_columns = st.columns(2, gap="medium")
                with video_columns[0]:
                    st.markdown("#### Video tải lên")
                    st.video(
                        video_bytes,
                        format=upload_mime or "video/mp4",
                        width="stretch",
                    )
                    st.caption("Video gốc chưa gắn thông tin nhận dạng.")
                with video_columns[1]:
                    st.markdown("#### Video nhận dạng và xác suất")
                    if analysis_is_current:
                        st.video(
                            analysis["processed_video"],
                            format="video/mp4",
                            width="stretch",
                        )
                        st.caption(
                            "Khung xanh: tỉnh táo · khung đỏ: vượt ngưỡng cảnh báo."
                        )
                        st.download_button(
                            "Tải video nhận dạng",
                            analysis["processed_video"],
                            "video_nhan_dang_buon_ngu.mp4",
                            "video/mp4",
                            icon=":material/download:",
                        )
                    else:
                        st.info(
                            "Nhấn **Phân tích video và tách ảnh cảnh báo** để tạo "
                            "video nhận dạng có xác suất."
                        )

            if analysis_is_current:
                snapshots = analysis["alert_snapshots"]
                points = analysis["probability_points"]

                with st.container(border=True):
                    st.markdown("### :material/analytics: Tổng quan kết quả")
                    if analysis["alert_frames"]:
                        st.error(
                            f"Phát hiện {analysis['alert_frames']:,} mẫu vượt ngưỡng "
                            f"{threshold:.0%}; đã tách {len(snapshots):,} ảnh đại diện."
                        )
                    elif analysis["face_frames"]:
                        st.success("Không có mẫu nào vượt ngưỡng cảnh báo đã chọn.")
                    else:
                        st.warning("Không phát hiện khuôn mặt trong các mẫu đã phân tích.")

                    metric_columns = st.columns(4)
                    metric_columns[0].metric(
                        "Thời lượng",
                        format_video_time(analysis["duration_seconds"]),
                    )
                    metric_columns[1].metric(
                        "Mẫu đã phân tích", f"{analysis['analyzed_frames']:,}"
                    )
                    metric_columns[2].metric(
                        "Mẫu vượt ngưỡng", f"{analysis['alert_frames']:,}"
                    )
                    alert_rate = (
                        analysis["alert_frames"] / analysis["face_frames"]
                        if analysis["face_frames"]
                        else 0.0
                    )
                    metric_columns[3].metric("Tỷ lệ cảnh báo", f"{alert_rate:.1%}")

                    if points:
                        max_chart_points = 2_000
                        point_stride = max(1, len(points) // max_chart_points)
                        chart_points = points[::point_stride]
                        chart_data = pd.DataFrame(
                            chart_points,
                            columns=["Thời gian (giây)", "Xác suất buồn ngủ"],
                        )
                        chart_data["Ngưỡng cảnh báo"] = threshold
                        st.markdown("#### Diễn biến xác suất theo thời gian")
                        st.line_chart(
                            chart_data,
                            x="Thời gian (giây)",
                            y=["Xác suất buồn ngủ", "Ngưỡng cảnh báo"],
                            y_label="Xác suất",
                            height=320,
                        )

                if snapshots:
                    with st.container(border=True):
                        st.markdown("### :material/warning: Ảnh cảnh báo đã tách")
                        st.caption(
                            "Mỗi ảnh được gắn thời điểm và xác suất. Khoảng cách lấy ảnh "
                            "giúp loại bớt các khung hình gần như trùng nhau."
                        )

                        alert_table = pd.DataFrame(
                            [
                                {
                                    "STT": index,
                                    "Khung hình": alert["frame_number"],
                                    "Thời điểm": format_video_time(
                                        alert["time_seconds"]
                                    ),
                                    "Xác suất": alert["probability"],
                                }
                                for index, alert in enumerate(snapshots, start=1)
                            ]
                        )
                        st.dataframe(
                            alert_table,
                            hide_index=True,
                            column_config={
                                "Xác suất": st.column_config.ProgressColumn(
                                    "Xác suất buồn ngủ",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="percent",
                                )
                            },
                            key="alert_table",
                        )

                        archive = build_alert_archive(snapshots, threshold)
                        st.download_button(
                            "Tải toàn bộ ảnh và báo cáo CSV",
                            archive,
                            "anh_canh_bao_buon_ngu.zip",
                            "application/zip",
                            icon=":material/download:",
                            type="primary",
                        )

                        st.markdown("#### Thư viện ảnh")
                        gallery_columns = st.columns(3)
                        for index, alert in enumerate(snapshots):
                            with gallery_columns[index % 3]:
                                with st.container(border=True):
                                    st.image(
                                        alert["image"],
                                        caption=(
                                            f"#{index + 1:02d} · "
                                            f"{format_video_time(alert['time_seconds'])} · "
                                            f"{alert['probability']:.1%}"
                                        ),
                                    )

st.divider()
st.markdown("#### :material/info: Phạm vi sử dụng")
st.caption(
    "Nhãn cảnh báo biểu thị xác suất của mô hình, không phải chẩn đoán y tế. "
    "Khi có nhiều người, hệ thống chỉ phân tích khuôn mặt lớn nhất. Video tải lên "
    "tối đa 200 MB; thời gian xử lý phụ thuộc độ dài video và tần suất lấy mẫu."
)
