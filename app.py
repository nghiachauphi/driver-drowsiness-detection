"""Realtime driver-drowsiness detection with the E6 MobileNetV2 model."""
import faulthandler
import os

# Streamlit Community Cloud uses a shared, resource-limited Linux CPU. Configure
# TensorFlow before importing it to avoid native oneDNN/thread-pool crashes.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Native ML/video wheels can terminate Python before Streamlit can display an
# exception. This makes a future cloud log include the active Python stack.
faulthandler.enable(all_threads=True)
print("[startup] Python runtime initialized", flush=True)

import csv
import io
import json
from pathlib import Path
from functools import lru_cache
import threading
import tempfile
import time
import zipfile

print("[startup] Importing PyAV", flush=True)
import av
print("[startup] Imported PyAV", flush=True)
print("[startup] Importing OpenCV", flush=True)
import cv2
print("[startup] Imported OpenCV", flush=True)
import numpy as np
import pandas as pd
import streamlit as st
print("[startup] Importing TensorFlow", flush=True)
import tensorflow as tf
print(f"[startup] Imported TensorFlow {tf.__version__}", flush=True)
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from streamlit_webrtc.credentials import get_available_ice_servers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from drowsiness_temporal import TemporalDrowsinessTracker


ROOT = Path(__file__).resolve().parent
IMPROVED_MODEL_PATH = ROOT / "outputs" / "models" / "improved_mobilenetv2.keras"
IMPROVED_METADATA_PATH = ROOT / "outputs" / "results" / "improved_model_metadata.json"
FALLBACK_MODEL_PATH = ROOT / "outputs" / "models" / "E6_MobileNetV2_subject.keras"
VIDEO_INFERENCE_VERSION = 5


def resolve_model_artifact():
    """Use the candidate only after a full run passes an unseen-subject gate."""
    if IMPROVED_MODEL_PATH.exists() and IMPROVED_METADATA_PATH.exists():
        try:
            metadata = json.loads(IMPROVED_METADATA_PATH.read_text(encoding="utf-8"))
            image_metrics = metadata.get("image_metrics", {})
            subject_metrics = metadata.get("subject_macro_metrics", {})
            candidate_is_deployable = metadata.get(
                "deployable",
                image_metrics.get("roc_auc", 0.0) >= 0.55
                and image_metrics.get("balanced_accuracy", 0.0) >= 0.55
                and subject_metrics.get("balanced_accuracy", 0.0) >= 0.55,
            )
            if not metadata.get("quick_run", True) and candidate_is_deployable:
                return IMPROVED_MODEL_PATH, metadata
        except (OSError, ValueError, TypeError):
            pass
    return FALLBACK_MODEL_PATH, {
        "model_name": "MobileNetV2 E6 (mô hình cũ)",
        "input_size": 96,
        "recommended_threshold": 0.50,
        "quick_run": False,
    }


MODEL_PATH, MODEL_METADATA = resolve_model_artifact()


def contains_turn_server(ice_servers):
    """Return whether an ICE configuration contains a TURN relay URL."""
    for server in ice_servers:
        urls = server.get("urls", []) if isinstance(server, dict) else []
        if isinstance(urls, str):
            urls = [urls]
        if any(str(url).lower().startswith(("turn:", "turns:")) for url in urls):
            return True
    return False


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
        # Pillow's unscaled default is only about 10 px, which made labels tiny
        # on minimal Linux images such as Streamlit Community Cloud.
        return ImageFont.load_default(size=size)


def draw_unicode_text(image_bgr, text, position, color_bgr, font_size=24):
    """Draw UTF-8 text on a BGR OpenCV frame and return the updated frame."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    color_rgb = tuple(reversed(color_bgr))
    stroke_width = max(1, round(font_size / 20))
    draw.text(
        position,
        text,
        font=load_unicode_font(font_size),
        fill=color_rgb,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )
    return cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)


def fit_overlay_font_size(text, preferred_size, max_width, min_size=20):
    """Shrink a preferred overlay font only when the label would be clipped."""
    font = load_unicode_font(preferred_size)
    left, _, right, _ = font.getbbox(text)
    text_width = max(1, right - left)
    if text_width <= max_width:
        return preferred_size
    return max(min_size, int(preferred_size * max_width / text_width))

st.set_page_config(
    page_title="Phân tích trạng thái buồn ngủ",
    page_icon=":material/visibility:",
    layout="wide",
)

st.html(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .st-key-image-result-success [data-testid="stAlert"] p {
        font-size: 1.075rem;
        line-height: 1.55;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """
)


@st.cache_resource(show_spinner="Đang nạp mô hình MobileNetV2…")
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")
    # custom_objects is only needed by the old E6 Lambda preprocessing layer.
    print(f"[startup] Loading model: {model_path.name}", flush=True)
    loaded_model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
    )
    print(f"[startup] Loaded model: {model_path.name}", flush=True)
    return loaded_model


@st.cache_resource
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


@st.cache_resource
def load_eye_detector():
    """Load an independent detector used to confirm two visibly open eyes."""
    path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise FileNotFoundError(f"Không tìm thấy bộ phát hiện mắt: {path}")
    return detector


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(
        self,
        model,
        detector,
        eye_detector,
        threshold,
        min_closed_seconds,
        window_seconds,
        closed_ratio_threshold,
    ):
        self.model = model
        self.detector = detector
        self.eye_detector = eye_detector
        self.threshold = threshold
        self.tracker = TemporalDrowsinessTracker(
            threshold,
            min_closed_seconds=min_closed_seconds,
            window_seconds=window_seconds,
            closed_ratio_threshold=closed_ratio_threshold,
        )
        self.lock = threading.Lock()
        self.last_probability = None
        self.last_face_count = 0
        self.last_open_eye_count = 0
        self.last_decision = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        probability, face_count, face_box = predict_frame(
            image, self.model, self.detector
        )
        open_eye_count = count_open_eyes(image, face_box, self.eye_detector)
        timestamp = float(frame.time) if frame.time is not None else time.monotonic()
        decision = self.tracker.update(
            timestamp,
            probability,
            open_eye_count,
            face_detected=bool(face_count),
        )
        image = draw_prediction(
            image,
            probability,
            face_box,
            self.threshold,
            temporal_decision=decision,
        )

        with self.lock:
            self.last_probability = probability
            self.last_face_count = face_count
            self.last_open_eye_count = open_eye_count
            self.last_decision = decision
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def predict_frame(image_bgr, model, detector):
    """Return probability, face count, and the largest detected face box."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if not len(faces):
        return None, 0, None

    # Prefer the largest face when several people are in view.
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    # The training images are square face crops. Match that geometry at runtime
    # and keep a small context margin instead of stretching a rectangular box.
    center_x, center_y = x + w / 2, y + h / 2
    side = max(w, h) * 1.10
    crop_x1 = max(0, int(center_x - side / 2))
    crop_y1 = max(0, int(center_y - side / 2))
    crop_x2 = min(image_bgr.shape[1], int(center_x + side / 2))
    crop_y2 = min(image_bgr.shape[0], int(center_y + side / 2))
    face_rgb = cv2.cvtColor(
        image_bgr[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_BGR2RGB
    )
    input_size = int(model.input_shape[1])
    face_rgb = cv2.resize(face_rgb, (input_size, input_size))
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
    probability = float(model(batch, training=False).numpy()[0, 0])
    return probability, len(faces), (x, y, w, h)


def count_open_eyes(image_bgr, face_box, eye_detector):
    """Return how many sides of the upper face contain a detected open eye."""
    if face_box is None:
        return 0

    x, y, w, h = face_box
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    top = y + int(0.12 * h)
    bottom = y + int(0.62 * h)
    eye_region = gray[top:bottom, x:x + w]
    if eye_region.size == 0:
        return 0

    eye_region = cv2.equalizeHist(eye_region)
    min_eye = max(12, int(w * 0.10))
    max_eye = max(min_eye + 1, int(w * 0.40))
    eyes = eye_detector.detectMultiScale(
        eye_region,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(min_eye, min_eye),
        maxSize=(max_eye, max_eye),
    )

    # Counting left/right sides is more stable than trusting every Haar box,
    # because one eye can occasionally produce two overlapping detections.
    left_eye = any(ex + ew / 2 < w / 2 for ex, _, ew, _ in eyes)
    right_eye = any(ex + ew / 2 >= w / 2 for ex, _, ew, _ in eyes)
    return int(left_eye) + int(right_eye)


def draw_prediction(
    image_bgr,
    probability,
    face_box,
    threshold,
    force_awake=False,
    temporal_decision=None,
    overlay_scale=1.0,
):
    """Draw one prediction using a Vietnamese-capable Unicode font."""
    result = image_bgr.copy()
    image_height, image_width = result.shape[:2]
    base_font_size = max(
        32,
        min(48, round(min(image_height, image_width) * 0.04)),
    )
    preferred_font_size = max(20, min(80, round(base_font_size * overlay_scale)))
    if face_box is None:
        label = "Không phát hiện khuôn mặt"
        font_size = fit_overlay_font_size(
            label,
            preferred_font_size,
            max_width=image_width - 40,
        )
        return draw_unicode_text(
            result,
            label,
            (20, 16),
            (0, 165, 255),
            font_size=font_size,
        )

    x, y, w, h = face_box
    if temporal_decision is not None:
        drowsy = temporal_decision.alert
    else:
        drowsy = probability >= threshold and not force_awake
    if drowsy:
        label = "CẢNH BÁO: BUỒN NGỦ"
    elif force_awake or (
        temporal_decision is not None and temporal_decision.open_eye_count >= 2
    ):
        label = "TỈNH TÁO (PHÁT HIỆN MẮT MỞ)"
    elif temporal_decision is not None and temporal_decision.closed_evidence:
        label = "ĐANG THEO DÕI DẤU HIỆU"
    else:
        label = "TỈNH TÁO"
    tracking = (
        temporal_decision is not None
        and temporal_decision.closed_evidence
        and not drowsy
    )
    if drowsy:
        color = (0, 0, 255)
    elif tracking:
        color = (0, 200, 255)
    else:
        color = (0, 190, 0)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
    overlay_text = f"{label}: {probability:.1%}"
    font_size = fit_overlay_font_size(
        overlay_text,
        preferred_font_size,
        max_width=max(80, image_width - x - 16),
    )
    return draw_unicode_text(
        result,
        overlay_text,
        (x, max(6, y - font_size - 12)),
        color,
        font_size=font_size,
    )


def annotate_frame(image_bgr, model, detector, eye_detector, threshold):
    """Detect the largest face and draw its prediction on one BGR frame."""
    probability, face_count, face_box = predict_frame(image_bgr, model, detector)
    open_eye_count = count_open_eyes(image_bgr, face_box, eye_detector)
    result = draw_prediction(
        image_bgr,
        probability,
        face_box,
        threshold,
        force_awake=open_eye_count >= 2,
    )
    return result, probability, face_count, open_eye_count


def analyze_uploaded_video(
    video_bytes,
    suffix,
    model,
    detector,
    eye_detector,
    threshold,
    analysis_rate,
    min_closed_seconds,
    window_seconds,
    closed_ratio_threshold,
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
        tracker = TemporalDrowsinessTracker(
            threshold,
            min_closed_seconds=min_closed_seconds,
            window_seconds=window_seconds,
            closed_ratio_threshold=closed_ratio_threshold,
        )
        alert_snapshots = []
        probability_points = []
        total_frames = analyzed_frames = face_frames = alert_frames = 0
        last_snapshot_time = -alert_interval
        current_probability = None
        current_face_box = None
        current_open_eye_count = 0
        current_decision = None
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
                current_open_eye_count = count_open_eyes(
                    frame, current_face_box, eye_detector
                )
                current_decision = tracker.update(
                    timestamp,
                    current_probability,
                    current_open_eye_count,
                    face_detected=bool(face_count),
                )
                if face_count:
                    face_frames += 1
                    probability_points.append(
                        (
                            timestamp,
                            current_probability,
                            current_decision.rolling_closed_ratio,
                        )
                    )
                    if current_decision.alert:
                        alert_frames += 1
                        can_save = timestamp - last_snapshot_time >= alert_interval
                        if can_save and len(alert_snapshots) < max_alert_images:
                            annotated = draw_prediction(
                                frame,
                                current_probability,
                                current_face_box,
                                threshold,
                                temporal_decision=current_decision,
                                overlay_scale=1.8,
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
                                        "open_eye_count": current_open_eye_count,
                                        "closed_ratio": current_decision.rolling_closed_ratio,
                                        "closed_seconds": current_decision.consecutive_closed_seconds,
                                        "reason": current_decision.reason,
                                        "image": jpeg.tobytes(),
                                    }
                                )
                                last_snapshot_time = timestamp

            annotated_frame = draw_prediction(
                frame,
                current_probability,
                current_face_box,
                threshold,
                temporal_decision=current_decision,
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
            "temporal_settings": {
                "min_closed_seconds": min_closed_seconds,
                "window_seconds": window_seconds,
                "closed_ratio_threshold": closed_ratio_threshold,
            },
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
    writer.writerow(
        [
            "STT",
            "Tệp ảnh",
            "Khung hình",
            "Thời điểm (giây)",
            "Điểm mô hình",
            "Ngưỡng mô hình",
            "Mắt mở",
            "Tỷ lệ dấu hiệu nhắm mắt",
            "Nhắm liên tục (giây)",
            "Lý do cảnh báo",
        ]
    )

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
                    alert["open_eye_count"],
                    f"{alert['closed_ratio']:.6f}",
                    f"{alert['closed_seconds']:.2f}",
                    alert["reason"],
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


header_content, header_source = st.columns(
    [2.35, 1], gap="large", vertical_alignment="bottom"
)
with header_content:
    st.title("Phân tích trạng thái buồn ngủ của tài xế")
    st.caption(
        "Kết hợp MobileNetV2, phát hiện mắt mở và xác nhận "
        "dấu hiệu theo thời gian"
    )
    st.markdown(
        f":blue-badge[{MODEL_METADATA['model_name']}] "
        f":gray-badge[Đầu vào {MODEL_METADATA['input_size']} × "
        f"{MODEL_METADATA['input_size']} px] "
        ":orange-badge[Mô hình thử nghiệm]"
    )
with header_source:
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
    recommended_threshold = float(
        MODEL_METADATA.get("recommended_threshold", 0.50)
    )
    recommended_threshold = min(0.90, max(0.10, recommended_threshold))
    recommended_threshold = round(recommended_threshold / 0.05) * 0.05
    threshold = st.slider(
        "Ngưỡng điểm mô hình",
        min_value=0.10,
        max_value=0.90,
        value=recommended_threshold,
        step=0.05,
        format="%.2f",
        help=(
            "Đây chỉ là điều kiện đầu tiên. Video chỉ cảnh báo sau khi dấu hiệu "
            "được duy trì đủ lâu hoặc xuất hiện với tỷ lệ cao trong cửa sổ thời gian. "
            "Điểm từ 90% sẽ cảnh báo đỏ ngay nếu không phát hiện đủ hai mắt mở."
        ),
    )
    st.metric("Ngưỡng mô hình đang dùng", f"{threshold:.0%}")

    st.markdown("#### :material/timeline: Xác nhận theo thời gian")
    min_closed_seconds = st.slider(
        "Thời gian dấu hiệu liên tục (giây)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.25,
        help="Không cảnh báo từ một khung hình đơn lẻ.",
    )
    window_seconds = st.slider(
        "Cửa sổ theo dõi (giây)",
        min_value=3.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
    )
    closed_ratio_threshold = st.slider(
        "Tỷ lệ dấu hiệu nhắm mắt trong cửa sổ",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
        format="%.2f",
    )

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
    print("[startup] Initializing inference resources", flush=True)
    model = load_model(str(MODEL_PATH))
    detector = load_detector()
    eye_detector = load_eye_detector()
    print("[startup] Inference resources ready", flush=True)
except Exception as exc:
    st.error(f"Không thể nạp mô hình: {exc}", icon=":material/error:")
    st.stop()

st.session_state.setdefault("video_analysis", None)
st.session_state.setdefault("video_analysis_key", None)

if mode == "Webcam":
    with st.container(border=True):
        st.markdown("### :material/videocam: Phân tích trực tiếp")
        st.info(
            "Bật webcam, cấp quyền camera cho trang này, sau đó nhấn **START**. "
            "Ảnh cảnh báo tách riêng hiện áp dụng cho video tải lên."
        )
        turn_configured = bool(os.getenv("HF_TOKEN"))
        ice_servers = get_available_ice_servers()
        turn_ready = contains_turn_server(ice_servers)
        if turn_ready:
            st.success(
                "TURN đã sẵn sàng cho kết nối webcam trên Cloud.",
                icon=":material/cloud_done:",
            )
        elif turn_configured:
            st.error(
                "Đã nhận `HF_TOKEN` nhưng không lấy được máy chủ TURN. "
                "Hãy kiểm tra token, xem log Cloud và reboot app.",
                icon=":material/cloud_off:",
            )
        else:
            st.caption(
                ":material/info: Bản Streamlit Cloud hiện chỉ có STUN. "
                "Nếu kết nối bị chờ lâu, hãy thêm `HF_TOKEN` vào "
                "**Manage app → Settings → Secrets** để bật TURN."
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
                    model,
                    detector,
                    eye_detector,
                    threshold,
                    min_closed_seconds,
                    window_seconds,
                    closed_ratio_threshold,
                ),
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={
                    "iceServers": ice_servers,
                },
                async_processing=True,
            )
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    probability = ctx.video_processor.last_probability
                    face_count = ctx.video_processor.last_face_count
                    open_eye_count = ctx.video_processor.last_open_eye_count
                    decision = ctx.video_processor.last_decision
                metric_columns = st.columns(4)
                metric_columns[0].metric("Khuôn mặt", face_count)
                metric_columns[1].metric(
                    "Điểm mô hình",
                    "—" if probability is None else f"{probability:.1%}",
                )
                metric_columns[2].metric("Mắt mở", open_eye_count)
                metric_columns[3].metric(
                    "Tỷ lệ dấu hiệu",
                    "—" if decision is None else f"{decision.rolling_closed_ratio:.1%}",
                )
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
                annotated, probability, face_count, open_eye_count = annotate_frame(
                    image, model, detector, eye_detector, threshold
                )
                awake_by_eyes = open_eye_count >= 2
                with st.container(border=True):
                    st.markdown("### :material/image_search: Kết quả phân tích ảnh")
                    if probability is None:
                        st.warning("Không phát hiện khuôn mặt trong ảnh.")
                    elif probability >= threshold and not awake_by_eyes:
                        st.error(
                            f"Cảnh báo buồn ngủ — điểm mô hình {probability:.1%} "
                            f"vượt ngưỡng {threshold:.0%}."
                        )
                    elif awake_by_eyes:
                        with st.container(key="image-result-success"):
                            st.success(
                                f"Phát hiện đủ hai mắt mở — kết luận **tỉnh táo**. "
                                f"Điểm thô MobileNetV2 là {probability:.1%} nhưng đã bị "
                                "bộ kiểm tra mắt mở bác bỏ."
                            )
                    else:
                        with st.container(key="image-result-success"):
                            st.success(
                                f"Chưa vượt ngưỡng cảnh báo — "
                                f"điểm mô hình {probability:.1%}."
                            )

                    metric_columns = st.columns(4)
                    metric_columns[0].metric("Khuôn mặt", face_count)
                    metric_columns[1].metric(
                        "Điểm mô hình",
                        "—" if probability is None else f"{probability:.1%}",
                    )
                    metric_columns[2].metric("Mắt mở", open_eye_count)
                    metric_columns[3].metric("Ngưỡng cảnh báo", f"{threshold:.0%}")

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
                VIDEO_INFERENCE_VERSION,
                upload.name,
                upload.size,
                threshold,
                analysis_rate,
                min_closed_seconds,
                window_seconds,
                closed_ratio_threshold,
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
                    f"{alert_interval:g} giây · xác nhận liên tục {min_closed_seconds:g} giây · "
                    f"cửa sổ {window_seconds:g} giây"
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
                        eye_detector,
                        threshold,
                        analysis_rate,
                        min_closed_seconds,
                        window_seconds,
                        closed_ratio_threshold,
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
                    st.markdown("#### Video nhận dạng và điểm mô hình")
                    if analysis_is_current:
                        st.video(
                            analysis["processed_video"],
                            format="video/mp4",
                            width="stretch",
                        )
                        st.caption(
                            "Khung vàng: đang theo dõi · khung đỏ: dấu hiệu đã kéo dài "
                            "đủ điều kiện cảnh báo; điểm từ 90% chuyển đỏ ngay nếu "
                            "không phát hiện đủ hai mắt mở."
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
                            "video nhận dạng có điểm mô hình."
                        )

            if analysis_is_current:
                snapshots = analysis["alert_snapshots"]
                points = analysis["probability_points"]

                with st.container(border=True):
                    st.markdown("### :material/analytics: Tổng quan kết quả")
                    if analysis["alert_frames"]:
                        st.error(
                            f"Phát hiện {analysis['alert_frames']:,} mẫu đã thỏa điều kiện "
                            f"thời gian; đã tách {len(snapshots):,} ảnh đại diện."
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
                        "Mẫu cảnh báo thời gian", f"{analysis['alert_frames']:,}"
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
                            columns=[
                                "Thời gian (giây)",
                                "Điểm mô hình",
                                "Tỷ lệ dấu hiệu nhắm mắt",
                            ],
                        )
                        chart_data["Ngưỡng mô hình"] = threshold
                        chart_data["Ngưỡng tỷ lệ"] = closed_ratio_threshold
                        st.markdown("#### Diễn biến tín hiệu theo thời gian")
                        st.line_chart(
                            chart_data,
                            x="Thời gian (giây)",
                            y=[
                                "Điểm mô hình",
                                "Tỷ lệ dấu hiệu nhắm mắt",
                                "Ngưỡng mô hình",
                                "Ngưỡng tỷ lệ",
                            ],
                            y_label="Điểm / tỷ lệ",
                            height=320,
                        )

                with st.container(border=True):
                    st.markdown("### :material/photo_library: Kết quả ảnh buồn ngủ")
                    st.caption(
                        "Các khung hình đã thỏa điều kiện cảnh báo theo thời gian "
                        "được tách từ video và hiển thị ngay bên dưới biểu đồ."
                    )

                    if not snapshots:
                        st.info(
                            "Chưa có khung hình nào thỏa điều kiện để tách ảnh. "
                            "Bạn có thể điều chỉnh ngưỡng hoặc thời gian xác nhận rồi phân tích lại."
                        )
                    else:
                        gallery_columns = st.columns(3, gap="medium")
                        for index, alert in enumerate(snapshots):
                            with gallery_columns[index % 3]:
                                with st.container(border=True):
                                    st.image(
                                        alert["image"],
                                        caption=(
                                            f"#{index + 1:02d} · "
                                            f"{format_video_time(alert['time_seconds'])} · "
                                            f"điểm {alert['probability']:.1%} · "
                                            f"tỷ lệ {alert['closed_ratio']:.1%}"
                                        ),
                                        width="stretch",
                                    )

                        st.divider()
                        st.markdown("#### Chi tiết và tải kết quả")
                        alert_table = pd.DataFrame(
                            [
                                {
                                    "STT": index,
                                    "Khung hình": alert["frame_number"],
                                    "Thời điểm": format_video_time(
                                        alert["time_seconds"]
                                    ),
                                    "Điểm mô hình": alert["probability"],
                                    "Mắt mở": alert["open_eye_count"],
                                    "Tỷ lệ nhắm": alert["closed_ratio"],
                                    "Nhắm liên tục": alert["closed_seconds"],
                                    "Lý do": alert["reason"],
                                }
                                for index, alert in enumerate(snapshots, start=1)
                            ]
                        )
                        st.dataframe(
                            alert_table,
                            hide_index=True,
                            column_config={
                                "Điểm mô hình": st.column_config.ProgressColumn(
                                    "Điểm mô hình",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="percent",
                                ),
                                "Tỷ lệ nhắm": st.column_config.ProgressColumn(
                                    "Tỷ lệ dấu hiệu nhắm mắt",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="percent",
                                ),
                                "Nhắm liên tục": st.column_config.NumberColumn(
                                    "Nhắm liên tục (giây)", format="%.2f"
                                ),
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

st.divider()
st.markdown("#### :material/info: Phạm vi sử dụng")
st.caption(
    "Điểm mô hình không phải xác suất đã được hiệu chỉnh hay chẩn đoán y tế. "
    "Cảnh báo video chỉ xuất hiện sau khi tín hiệu được xác nhận theo thời gian. "
    "Khi có nhiều người, hệ thống chỉ phân tích khuôn mặt lớn nhất. Video tải lên "
    "tối đa 200 MB; thời gian xử lý phụ thuộc độ dài video và tần suất lấy mẫu."
)
