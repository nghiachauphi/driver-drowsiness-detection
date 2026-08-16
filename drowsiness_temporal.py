"""Temporal decision logic shared by uploaded-video and webcam inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalDecision:
    """State returned after adding one analyzed video frame."""

    alert: bool
    reason: str
    model_score: float | None
    open_eye_count: int
    closed_evidence: bool
    high_confidence_evidence: bool
    consecutive_closed_seconds: float
    rolling_closed_ratio: float
    observed_window_seconds: float


class TemporalDrowsinessTracker:
    """Require persistent closed-eye evidence before raising an alert.

    The CNN score alone is intentionally insufficient. A frame contributes
    closed-eye evidence only when the CNN crosses its threshold and the
    independent eye detector cannot confirm two open eyes.
    """

    def __init__(
        self,
        model_threshold: float,
        min_closed_seconds: float = 1.5,
        window_seconds: float = 10.0,
        closed_ratio_threshold: float = 0.40,
        min_window_seconds: float | None = None,
        high_score_threshold: float = 0.90,
        high_score_seconds: float = 0.0,
        interruption_tolerance_seconds: float = 0.35,
    ) -> None:
        self.model_threshold = float(model_threshold)
        self.min_closed_seconds = float(min_closed_seconds)
        self.window_seconds = float(window_seconds)
        self.closed_ratio_threshold = float(closed_ratio_threshold)
        if min_window_seconds is None:
            min_window_seconds = min(1.5, self.min_closed_seconds)
        self.min_window_seconds = min(float(min_window_seconds), self.window_seconds)
        self.high_score_threshold = max(
            self.model_threshold, float(high_score_threshold)
        )
        self.high_score_seconds = min(
            float(high_score_seconds), self.min_closed_seconds
        )
        self.interruption_tolerance_seconds = float(
            interruption_tolerance_seconds
        )
        self._history: deque[tuple[float, bool]] = deque()
        self._last_timestamp: float | None = None
        self._last_closed = False
        self._last_high_confidence = False
        self._interruption_seconds = 0.0
        self._consecutive_closed_seconds = 0.0
        self._consecutive_high_score_seconds = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._last_timestamp = None
        self._last_closed = False
        self._last_high_confidence = False
        self._interruption_seconds = 0.0
        self._consecutive_closed_seconds = 0.0
        self._consecutive_high_score_seconds = 0.0

    def update(
        self,
        timestamp: float,
        model_score: float | None,
        open_eye_count: int,
        face_detected: bool,
    ) -> TemporalDecision:
        timestamp = float(timestamp)
        open_eye_count = int(open_eye_count)
        model_positive = (
            face_detected
            and model_score is not None
            and float(model_score) >= self.model_threshold
        )
        eyes_confirmed_open = open_eye_count >= 2
        closed_evidence = bool(model_positive and not eyes_confirmed_open)
        high_confidence_evidence = bool(
            closed_evidence and float(model_score) >= self.high_score_threshold
        )

        if self._last_timestamp is None or timestamp < self._last_timestamp:
            delta = 0.0
        else:
            # Avoid treating a long gap without analyzed frames as continuous closure.
            delta = min(timestamp - self._last_timestamp, 1.0)

        if eyes_confirmed_open:
            # A reliable observation of both eyes open is stronger than a noisy
            # CNN score and must cancel the pending alert immediately.
            self._interruption_seconds = 0.0
            self._consecutive_closed_seconds = 0.0
            self._consecutive_high_score_seconds = 0.0
        elif closed_evidence:
            self._interruption_seconds = 0.0
            if self._last_closed:
                self._consecutive_closed_seconds += delta
            if high_confidence_evidence:
                if self._last_high_confidence:
                    self._consecutive_high_score_seconds += delta
            else:
                self._consecutive_high_score_seconds = 0.0
        else:
            # Haar detection can miss one sampled frame during a head movement
            # or a yawn. Preserve the pending sequence for a very short gap,
            # but do not add that gap to the accumulated evidence time.
            self._interruption_seconds += delta
            if self._interruption_seconds > self.interruption_tolerance_seconds:
                self._consecutive_closed_seconds = 0.0
                self._consecutive_high_score_seconds = 0.0

        self._last_timestamp = timestamp
        self._last_closed = closed_evidence
        self._last_high_confidence = high_confidence_evidence

        if face_detected:
            self._history.append((timestamp, closed_evidence))
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if self._history:
            rolling_closed_ratio = sum(value for _, value in self._history) / len(
                self._history
            )
            observed_window_seconds = self._history[-1][0] - self._history[0][0]
        else:
            rolling_closed_ratio = 0.0
            observed_window_seconds = 0.0

        prolonged_closure = (
            self._consecutive_closed_seconds >= self.min_closed_seconds
        )
        sustained_high_score = (
            high_confidence_evidence
            and self._consecutive_high_score_seconds >= self.high_score_seconds
        )
        high_closed_ratio = (
            observed_window_seconds >= self.min_window_seconds
            and rolling_closed_ratio >= self.closed_ratio_threshold
        )
        alert = bool(
            closed_evidence
            and (sustained_high_score or prolonged_closure or high_closed_ratio)
        )

        if sustained_high_score:
            reason = "Điểm mô hình rất cao và không phát hiện đủ hai mắt mở"
        elif prolonged_closure:
            reason = (
                f"Mắt có dấu hiệu nhắm liên tục "
                f"{self._consecutive_closed_seconds:.1f} giây"
            )
        elif high_closed_ratio:
            reason = (
                f"Tỷ lệ dấu hiệu nhắm mắt {rolling_closed_ratio:.0%} "
                f"trong {observed_window_seconds:.1f} giây"
            )
        elif eyes_confirmed_open:
            reason = "Phát hiện đủ hai mắt mở"
        elif not face_detected:
            reason = "Không phát hiện khuôn mặt"
        elif model_positive:
            reason = "Đang chờ tín hiệu duy trì đủ lâu"
        else:
            reason = "Chưa có dấu hiệu buồn ngủ kéo dài"

        return TemporalDecision(
            alert=alert,
            reason=reason,
            model_score=None if model_score is None else float(model_score),
            open_eye_count=open_eye_count,
            closed_evidence=closed_evidence,
            high_confidence_evidence=high_confidence_evidence,
            consecutive_closed_seconds=self._consecutive_closed_seconds,
            rolling_closed_ratio=rolling_closed_ratio,
            observed_window_seconds=observed_window_seconds,
        )
