import numpy as np
import pytest

from core.schemas import ModalitySource
from core.vision_processor import VisionProcessor, compute_ear


@pytest.fixture
def dummy_config():
    return {
        "vision": {
            "models": {
                "face_landmarker_path": "models/face_landmarker.task",
                "pose_landmarker_path": "models/pose_landmarker_lite.task",
            },
            "calibration": {"min_samples": 5},
            "eyes": {
                "default_ear_threshold": 0.20,
                "min_blink_duration_ms": 50,
                "drowsy_blink_duration_ms": 350,
                "rolling_windows_seconds": {"short": 30, "long": 300},
                "perclos_drowsy_threshold": 0.15,
            },
            "posture": {
                "angle_tolerance_deg": 10.0,
            },
            "quality": {
                "min_brightness": 40.0,
                "max_brightness": 220.0,
                "blur_laplacian_threshold": 50.0,
            }
        }
    }


def test_compute_ear_geometry():
    pts = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [2.0, -1.0],
        [1.0, -1.0]
    ])
    ear = compute_ear(pts)
    assert ear > 0.6

    pts_closed = pts.copy()
    pts_closed[:, 1] = 0.0
    ear_closed = compute_ear(pts_closed)
    assert ear_closed == pytest.approx(0.0, abs=1e-5)


def test_frame_quality_metrics(dummy_config, monkeypatch):
    monkeypatch.setattr(VisionProcessor, "_init_models", lambda self: None)
    vp = VisionProcessor(dummy_config)

    black = np.zeros((100, 100, 3), dtype=np.uint8)
    q_black, m_black = vp.assess_frame_quality(black)
    assert q_black < 0.2
    assert m_black["brightness"] == 0.0

    normal = np.full((100, 100, 3), 128, dtype=np.uint8)
    normal[::2, ::2] = 255
    q_normal, m_normal = vp.assess_frame_quality(normal)
    assert q_normal > 0.7


def test_posture_degradation(dummy_config, monkeypatch):
    monkeypatch.setattr(VisionProcessor, "_init_models", lambda self: None)
    vp = VisionProcessor(dummy_config)

    vp.baselines["posture_angle_deg"] = 90.0
    vp.baselines["head_shoulder_ratio"] = 1.0

    good_pose = np.zeros((33, 3))
    good_pose[0] = [100.0, 50.0, 0.0]
    good_pose[7] = [90.0, 60.0, 0.0]
    good_pose[8] = [110.0, 60.0, 0.0]
    good_pose[11] = [50.0, 150.0, 0.0]
    good_pose[12] = [150.0, 150.0, 0.0]

    score_good = vp._process_posture(good_pose)
    assert score_good >= 0.90

    slouched_pose = good_pose.copy()
    slouched_pose[0] = [100.0, 130.0, 0.0]
    slouched_pose[7] = [90.0, 135.0, 0.0]
    slouched_pose[8] = [110.0, 135.0, 0.0]

    score_bad = vp._process_posture(slouched_pose)
    assert score_bad < score_good


def test_blink_dynamics_state_machine(dummy_config, monkeypatch):
    monkeypatch.setattr(VisionProcessor, "_init_models", lambda self: None)
    vp = VisionProcessor(dummy_config)
    vp.baselines["ear_threshold"] = 0.20

    lm = np.zeros((478, 3))
    for idx in VisionProcessor.LEFT_EYE_INDICES + VisionProcessor.RIGHT_EYE_INDICES:
        lm[idx] = [10.0, 10.0, 0.0]
    lm[33] = [0.0, 0.0, 0.0]; lm[160] = [1.0, 1.0, 0.0]; lm[158] = [2.0, 1.0, 0.0]
    lm[133] = [3.0, 0.0, 0.0]; lm[153] = [2.0, -1.0, 0.0]; lm[144] = [1.0, -1.0, 0.0]
    lm[362] = [0.0, 0.0, 0.0]; lm[385] = [1.0, 1.0, 0.0]; lm[387] = [2.0, 1.0, 0.0]
    lm[263] = [3.0, 0.0, 0.0]; lm[373] = [2.0, -1.0, 0.0]; lm[380] = [1.0, -1.0, 0.0]

    ear, bpm, mean_dur, perclos = vp._process_eyes_and_blinks(lm, now=100.0)
    assert ear > 0.20
    assert not vp._is_blinking

    lm_closed = lm.copy()
    for idx in [160, 158, 153, 144, 385, 387, 373, 380]:
        lm_closed[idx, 1] = 0.0

    ear, bpm, mean_dur, perclos = vp._process_eyes_and_blinks(lm_closed, now=100.05)
    assert ear < 0.20
    assert vp._is_blinking

    ear, bpm, mean_dur, perclos = vp._process_eyes_and_blinks(lm, now=100.20)
    assert not vp._is_blinking
    assert len(vp.blink_durations) == 1
    assert vp.blink_durations[0] == pytest.approx(150.0, abs=1.0)
