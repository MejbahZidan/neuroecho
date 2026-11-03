from collections import deque
import logging
import math
import os
import time
from typing import Deque, Dict, List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from core.schemas import ModalityReading, ModalitySource

logger = logging.getLogger(__name__)


def compute_ear(eye_landmarks: np.ndarray) -> float:
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


class VisionProcessor:
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    TRACKED_BLENDSHAPES = [
        "browDownLeft", "browDownRight", "browInnerUp",
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "eyeSquintLeft", "eyeSquintRight", "jawOpen"
    ]

    def __init__(self, config: dict):
        self.config = config
        self.v_cfg = config.get("vision", {})
        
        self.calibrating = False
        self.calibrated = False
        self.calibration_samples: Dict[str, List[float]] = {
            "ear": [],
            "posture_angle": [],
            "head_shoulder_ratio": [],
        }
        self.baselines: Dict[str, float] = {
            "ear_threshold": self.v_cfg.get("eyes", {}).get("default_ear_threshold", 0.21),
            "posture_angle_deg": 90.0,
            "head_shoulder_ratio": 1.0,
        }
        self.blendshape_baselines: Dict[str, Tuple[float, float]] = {}

        self.ear_history: Deque[Tuple[float, float, bool]] = deque()
        self.blink_durations: Deque[float] = deque()
        self.blink_timestamps: Deque[float] = deque()
        self._is_blinking = False
        self._blink_start_time = 0.0

        self._frame_count = 0
        self._cached_posture_score = 1.0
        self._cached_blendshapes: Dict[str, float] = {}

        self._init_models()

    def _ensure_model_exists(self, url: str, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info("Downloading required model to %s...", path)
            urllib.request.urlretrieve(url, path)

    def _init_models(self) -> None:
        models_cfg = self.v_cfg.get("models", {})
        face_path = models_cfg.get("face_landmarker_path", "models/face_landmarker.task")
        face_url = models_cfg.get("face_landmarker_url", "")
        pose_path = models_cfg.get("pose_landmarker_path", "models/pose_landmarker_lite.task")
        pose_url = models_cfg.get("pose_landmarker_url", "")

        self._ensure_model_exists(face_url, face_path)
        self._ensure_model_exists(pose_url, pose_path)

        base_options_face = mp_python.BaseOptions(model_asset_path=face_path)
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

        base_options_pose = mp_python.BaseOptions(model_asset_path=pose_path)
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

    def start_calibration(self) -> None:
        self.calibration_samples = {"ear": [], "posture_angle": [], "head_shoulder_ratio": []}
        for shape in self.TRACKED_BLENDSHAPES:
            self.calibration_samples[shape] = []
        self.calibrating = True
        self.calibrated = False

    def finish_calibration(self) -> None:
        if len(self.calibration_samples["ear"]) < self.v_cfg.get("calibration", {}).get("min_samples", 100):
            logger.warning("Insufficient samples for robust calibration; using defaults.")
            self.calibrated = True
            self.calibrating = False
            return

        mean_ear = float(np.mean(self.calibration_samples["ear"]))
        self.baselines["ear_threshold"] = max(0.15, mean_ear * 0.75)
        self.baselines["posture_angle_deg"] = float(np.mean(self.calibration_samples["posture_angle"]))
        self.baselines["head_shoulder_ratio"] = float(np.mean(self.calibration_samples["head_shoulder_ratio"]))

        for shape in self.TRACKED_BLENDSHAPES:
            vals = self.calibration_samples[shape]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 1.0
            self.blendshape_baselines[shape] = (mean, max(std, 1e-4))

        self.calibrated = True
        self.calibrating = False
        logger.info("Calibration locked. EAR threshold: %.3f", self.baselines["ear_threshold"])

    def assess_frame_quality(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        q_cfg = self.v_cfg.get("quality", {})
        min_b = q_cfg.get("min_brightness", 40.0)
        max_b = q_cfg.get("max_brightness", 220.0)
        blur_thresh = q_cfg.get("blur_laplacian_threshold", 100.0)

        if brightness < min_b:
            b_score = max(0.0, brightness / min_b)
        elif brightness > max_b:
            b_score = max(0.0, (255.0 - brightness) / (255.0 - max_b))
        else:
            b_score = 1.0

        blur_score = min(1.0, blur_var / blur_thresh)
        quality = float(0.5 * b_score + 0.5 * blur_score)
        return quality, {"brightness": brightness, "blur": blur_var}

    def _process_eyes_and_blinks(
        self,
        landmarks: np.ndarray,
        now: float
    ) -> Tuple[float, float, float, float]:
        left_pts = landmarks[self.LEFT_EYE_INDICES, :2]
        right_pts = landmarks[self.RIGHT_EYE_INDICES, :2]
        ear = (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0

        is_closed = ear < self.baselines["ear_threshold"]
        self.ear_history.append((now, ear, is_closed))

        cutoff_long = now - self.v_cfg.get("eyes", {}).get("rolling_windows_seconds", {}).get("long", 300)
        while self.ear_history and self.ear_history[0][0] < cutoff_long:
            self.ear_history.popleft()

        min_duration = self.v_cfg.get("eyes", {}).get("min_blink_duration_ms", 80)
        if is_closed:
            if not self._is_blinking:
                self._is_blinking = True
                self._blink_start_time = now
        else:
            if self._is_blinking:
                duration_ms = (now - self._blink_start_time) * 1000.0
                self._is_blinking = False
                if duration_ms >= min_duration:
                    self.blink_durations.append(duration_ms)
                    self.blink_timestamps.append(now)

        cutoff_blink = now - 60.0
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff_blink:
            self.blink_timestamps.popleft()
            if self.blink_durations:
                self.blink_durations.popleft()

        blinks_per_min = float(len(self.blink_timestamps))
        mean_blink_dur = float(np.mean(self.blink_durations)) if self.blink_durations else 150.0

        cutoff_short = now - self.v_cfg.get("eyes", {}).get("rolling_windows_seconds", {}).get("short", 30)
        recent_samples = [item for item in self.ear_history if item[0] >= cutoff_short]
        perclos_30s = float(sum(1 for item in recent_samples if item[2]) / len(recent_samples)) if recent_samples else 0.0

        return ear, blinks_per_min, mean_blink_dur, perclos_30s

    def _process_posture(self, pose_landmarks: Optional[np.ndarray]) -> float:
        if pose_landmarks is None:
            return self._cached_posture_score

        nose = pose_landmarks[0, :2]
        left_ear, right_ear = pose_landmarks[7, :2], pose_landmarks[8, :2]
        left_shoulder, right_shoulder = pose_landmarks[11, :2], pose_landmarks[12, :2]

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        ear_center = (left_ear + right_ear) / 2.0

        dy = shoulder_center[1] - ear_center[1]
        dx = shoulder_center[0] - ear_center[0]
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))

        s_dy = abs(right_shoulder[1] - left_shoulder[1])
        s_dx = abs(right_shoulder[0] - left_shoulder[0]) + 1e-6
        shoulder_tilt = math.degrees(math.atan2(s_dy, s_dx))

        head_shoulder_dist = np.linalg.norm(shoulder_center - nose)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder) + 1e-6
        distance_ratio = float(head_shoulder_dist / shoulder_width)

        if self.calibrating:
            self.calibration_samples["posture_angle"].append(angle_deg)
            self.calibration_samples["head_shoulder_ratio"].append(distance_ratio)

        ref_angle = self.baselines["posture_angle_deg"]
        ref_ratio = self.baselines["head_shoulder_ratio"]
        tol = self.v_cfg.get("posture", {}).get("angle_tolerance_deg", 12.0)

        angle_err = max(0.0, abs(angle_deg - ref_angle) - tol) / 45.0
        tilt_err = max(0.0, shoulder_tilt - 5.0) / 30.0
        ratio_drop = max(0.0, (ref_ratio - distance_ratio) / ref_ratio)

        penalty = min(1.0, 0.4 * angle_err + 0.3 * tilt_err + 0.3 * ratio_drop)
        self._cached_posture_score = float(max(0.0, 1.0 - penalty))
        return self._cached_posture_score

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> ModalityReading:
        now = timestamp if timestamp is not None else time.time()
        self._frame_count += 1
        skip_n = self.v_cfg.get("frame_skip_expressions", 2)
        run_heavy = (self._frame_count % skip_n == 0)

        quality, q_metrics = self.assess_frame_quality(frame_bgr)
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        face_result = self.face_landmarker.detect(mp_image)

        if not face_result.face_landmarks:
            return ModalityReading(
                source=ModalitySource.VISION,
                timestamp=now,
                valence=0.0,
                arousal=0.0,
                fatigue=0.5,
                stress=0.0,
                confidence=0.0,
                quality=quality,
                features={"face_detected": False, **q_metrics}
            )

        face_landmarks_raw = face_result.face_landmarks[0]
        landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks_raw])

        ear, blinks_per_min, mean_blink_dur, perclos_30s = self._process_eyes_and_blinks(landmarks, now)

        if self.calibrating:
            self.calibration_samples["ear"].append(ear)

        if run_heavy and face_result.face_blendshapes:
            raw_shapes = {b.category_name: b.score for b in face_result.face_blendshapes[0]}
            for shape in self.TRACKED_BLENDSHAPES:
                val = raw_shapes.get(shape, 0.0)
                if self.calibrating:
                    self.calibration_samples[shape].append(val)
                if self.calibrated and shape in self.blendshape_baselines:
                    mean, std = self.blendshape_baselines[shape]
                    self._cached_blendshapes[shape] = (val - mean) / std
                else:
                    self._cached_blendshapes[shape] = val

        if run_heavy:
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                raw_pose = pose_result.pose_landmarks[0]
                pose_coords = np.array([[lm.x * w, lm.y * h, lm.z] for lm in raw_pose])
                posture_score = self._process_posture(pose_coords)
            else:
                posture_score = self._cached_posture_score
        else:
            posture_score = self._cached_posture_score

        smile = max(self._cached_blendshapes.get("mouthSmileLeft", 0.0), self._cached_blendshapes.get("mouthSmileRight", 0.0))
        frown = max(self._cached_blendshapes.get("mouthFrownLeft", 0.0), self._cached_blendshapes.get("mouthFrownRight", 0.0))
        brow_down = max(self._cached_blendshapes.get("browDownLeft", 0.0), self._cached_blendshapes.get("browDownRight", 0.0))

        valence = float(np.tanh(smile - frown))

        brow_stress = max(0.0, brow_down)
        blink_stress = max(0.0, (blinks_per_min - 25.0) / 20.0)
        posture_stress = 1.0 - posture_score
        stress = float(np.clip(0.4 * brow_stress + 0.3 * blink_stress + 0.3 * posture_stress, 0.0, 1.0))

        drowsy_dur = self.v_cfg.get("eyes", {}).get("drowsy_blink_duration_ms", 350)
        dur_fatigue = min(1.0, max(0.0, (mean_blink_dur - 200.0) / (drowsy_dur - 200.0)))
        perclos_fatigue = min(1.0, perclos_30s / self.v_cfg.get("eyes", {}).get("perclos_drowsy_threshold", 0.15))
        fatigue = float(np.clip(0.6 * perclos_fatigue + 0.4 * dur_fatigue, 0.0, 1.0))

        arousal = float(np.clip(0.5 * (1.0 - fatigue) + 0.5 * stress, 0.0, 1.0))

        calib_mult = 1.0 if self.calibrated else 0.6
        confidence = float(np.clip(quality * calib_mult, 0.0, 1.0))

        return ModalityReading(
            source=ModalitySource.VISION,
            timestamp=now,
            valence=valence,
            arousal=arousal,
            fatigue=fatigue,
            stress=stress,
            confidence=confidence,
            quality=quality,
            features={
                "face_detected": True,
                "ear": ear,
                "blinks_per_min": blinks_per_min,
                "mean_blink_duration_ms": mean_blink_dur,
                "perclos_30s": perclos_30s,
                "posture_ergonomic_score": posture_score,
                "blendshapes": dict(self._cached_blendshapes),
                **q_metrics
            }
        )
