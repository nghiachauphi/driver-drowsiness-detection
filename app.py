"""Realtime driver-drowsiness detection with the E6 MobileNetV2 model."""
from pathlib import Path
import threading
import tempfile

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "models" / "E6_MobileNetV2_subject.keras"
INPUT_SIZE = 96

st.set_page_config(page_title="Phát hiện buồn ngủ", page_icon="🚗", layout="wide")


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        probability = None

        if len(faces):
            # Prefer the largest face when several people are in view.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_rgb = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
            batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
            probability = float(self.model(batch, training=False).numpy()[0, 0])
            drowsy = probability >= self.threshold
            label = "BUỒN NGỦ" if drowsy else "TỈNH TÁO"
            color = (0, 0, 255) if drowsy else (0, 190, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(image, f"{label}: {probability:.1%}", (x, max(28, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Khong phat hien khuon mat", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)

        with self.lock:
            self.last_probability = probability
            self.last_face_count = len(faces)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def annotate_frame(image_bgr, model, detector, threshold):
    """Detect the largest face and draw its prediction on one BGR frame."""
    result = image_bgr.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    probability = None
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_rgb = cv2.cvtColor(result[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
        batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
        probability = float(model(batch, training=False).numpy()[0, 0])
        drowsy = probability >= threshold
        label = "BUỒN NGỦ" if drowsy else "TỈNH TÁO"
        color = (0, 0, 255) if drowsy else (0, 190, 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result, f"{label}: {probability:.1%}", (x, max(28, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(result, "Khong phat hien khuon mat", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
    return result, probability, len(faces)


def process_uploaded_video(video_bytes, suffix, model, detector, threshold):
    """Create a browser-compatible H.264 MP4 with annotations."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source_file:
        source_file.write(video_bytes)
        source_path = source_file.name
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        capture = cv2.VideoCapture(source_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Không đọc được dữ liệu video.")
        # Browsers commonly cannot play OpenCV's MP4V output. PyAV is
        # provided by streamlit-webrtc and writes H.264, which st.video supports.
        output = av.open(output_path, mode="w", format="mp4")
        stream = output.add_stream("libx264", rate=max(1, round(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        probabilities, frames = [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, probability, _ = annotate_frame(frame, model, detector, threshold)
            video_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
            if probability is not None:
                probabilities.append(probability)
            frames += 1
        capture.release()
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        if frames == 0:
            raise ValueError("Video không có khung hình để xử lý.")
        return Path(output_path).read_bytes(), frames, probabilities
    finally:
        Path(source_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


st.title("🚗 Phát hiện buồn ngủ")
st.caption("MobileNetV2 (E6) · webcam hoặc ảnh/video tải lên")

with st.sidebar:
    st.header("Thiết lập")
    threshold = st.slider("Ngưỡng cảnh báo buồn ngủ", 0.10, 0.90, 0.50, 0.05)
    st.caption("Xác suất ≥ ngưỡng sẽ hiển thị cảnh báo đỏ.")
    st.divider()
    st.warning("Mô hình này được huấn luyện ở chế độ pilot/CPU. Chỉ dùng để trình diễn, không dùng làm thiết bị an toàn thực tế.")

try:
    model = load_model()
    detector = load_detector()
except Exception as exc:
    st.error(f"Không thể nạp mô hình: {exc}")
    st.stop()

mode = st.segmented_control(
    "Nguồn đầu vào",
    ["Tải ảnh/video", "Webcam"],
    default="Tải ảnh/video",
    selection_mode="single",
    required=True,
)

if mode == "Webcam":
    st.info(
        "Kết nối và bật webcam trước, sau đó bật công tắc bên dưới và nhấn "
        "**START**. Hãy cấp quyền camera cho trang `localhost` khi trình duyệt hỏi."
    )
    st.caption(
        "Nếu trình phát báo `NotFoundError: Requested device not found`, hệ điều hành "
        "hoặc trình duyệt chưa nhận được camera. Có thể chuyển sang Tải ảnh/video để "
        "tiếp tục sử dụng ứng dụng."
    )
    camera_enabled = st.toggle(
        "Khởi tạo webcam",
        value=False,
        help="Chỉ bật sau khi webcam đã được kết nối và được Windows nhận.",
    )
    if camera_enabled:
        ctx = webrtc_streamer(
            key="drowsiness-camera",
            video_processor_factory=lambda: DrowsinessProcessor(model, detector, threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            left, right = st.columns(2)
            with ctx.video_processor.lock:
                prob = ctx.video_processor.last_probability
                face_count = ctx.video_processor.last_face_count
            left.metric("Khuôn mặt phát hiện", face_count)
            right.metric("Xác suất buồn ngủ", "—" if prob is None else f"{prob:.1%}")
    else:
        st.warning(
            "Webcam chưa được khởi tạo. Nếu máy không có camera, hãy dùng chế độ "
            "Tải ảnh/video."
        )
else:
    st.info("Tải lên một ảnh hoặc video. Video được xử lý từng khung hình và trả về bản đã gắn nhãn.")
    upload = st.file_uploader(
        "Chọn ảnh hoặc video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
        max_upload_size=200, key="media_upload"
    )
    if upload is not None:
        is_image = upload.type.startswith("image/") or upload.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        if is_image:
            raw = np.frombuffer(upload.getvalue(), np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Không thể đọc ảnh đã tải lên.")
            else:
                annotated, probability, face_count = annotate_frame(image, model, detector, threshold)
                left, right = st.columns(2)
                left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc")
                right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Kết quả nhận dạng")
                m1, m2 = st.columns(2)
                m1.metric("Khuôn mặt phát hiện", face_count)
                m2.metric("Xác suất buồn ngủ", "—" if probability is None else f"{probability:.1%}")
        else:
            st.subheader("Video đã tải lên")
            st.video(upload.getvalue(), format=upload.type or "video/mp4", width=720)
            if st.button("Nhận dạng video", type="primary", icon=":material/play_arrow:"):
                with st.spinner("Đang xử lý video…"):
                    try:
                        result, frames, probabilities = process_uploaded_video(
                            upload.getvalue(), Path(upload.name).suffix, model, detector, threshold
                        )
                        st.subheader("Video kết quả")
                        st.video(result, format="video/mp4", width=720)
                        left, right = st.columns(2)
                        left.metric("Số khung hình xử lý", frames)
                        right.metric(
                            "Xác suất buồn ngủ cao nhất", "—" if not probabilities else f"{max(probabilities):.1%}"
                        )
                        st.download_button("Tải video kết quả", result, "ket_qua_nhan_dang.mp4", "video/mp4",
                                           icon=":material/download:")
                    except Exception as exc:
                        st.error(f"Không thể xử lý video: {exc}")

st.markdown("""
### Lưu ý sử dụng

- Nhãn đỏ nghĩa là xác suất buồn ngủ vượt ngưỡng đã chọn; đây không phải chẩn đoán y tế.
- Khi có nhiều người, app chỉ nhận dạng khuôn mặt lớn nhất trong khung hình.
- Để dừng camera, nhấn **STOP** trên trình phát.
- Video tải lên tối đa 200 MB; video dài sẽ cần thời gian xử lý lâu hơn.
""")
