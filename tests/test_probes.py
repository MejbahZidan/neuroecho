"""Unit tests for cognitive probes (trigger logic, cooldowns, trial filtering, and CFI computation)."""

import pytest

from core.cognitive_probes import CognitiveProbeEngine, ProbeSessionResult, ProbeTriggerManager, ProbeType, TrialResult
from core.schemas import ModalitySource


@pytest.fixture
def dummy_config():
    return {
        "probes": {
            "trigger_threshold": 0.60,
            "consecutive_triggers": 3,
            "cooldown_minutes": 30,
            "probes_per_hour": 2,
        }
    }


def test_trigger_manager_consecutive_and_span_rules(dummy_config):
    manager = ProbeTriggerManager(dummy_config)

    # 1. High fatigue but only 1 reading -> False
    assert not manager.record_reading(fatigue=0.8, stress=0.2, confidence=0.7, timestamp=100.0)

    # 2. Second reading at t=130s -> False (only 2 readings)
    assert not manager.record_reading(fatigue=0.8, stress=0.2, confidence=0.7, timestamp=130.0)

    # 3. Third reading at t=140s (< 60s total span from t=100 to t=140) -> False
    assert not manager.record_reading(fatigue=0.8, stress=0.2, confidence=0.7, timestamp=140.0)

    # 4. Fourth reading at t=165s (>= 3 readings AND span 165 - 100 = 65s >= 60s) -> True!
    assert manager.record_reading(fatigue=0.8, stress=0.2, confidence=0.7, timestamp=165.0)


def test_trigger_manager_cooldown_and_snooze(dummy_config):
    manager = ProbeTriggerManager(dummy_config)

    # Mark a probe completed at t=1000s
    manager.mark_completed(timestamp=1000.0)

    # During 30-min cooldown (t=1000 to t=2800), triggers must be suppressed
    for t in [1100.0, 1200.0, 1300.0, 1400.0]:
        assert not manager.record_reading(fatigue=0.9, stress=0.9, confidence=0.9, timestamp=t)

    # Test explicit snooze
    manager._completed_timestamps.clear()  # reset cooldown
    manager.snooze(minutes=10, timestamp=3000.0)
    assert not manager.record_reading(fatigue=0.9, stress=0.9, confidence=0.9, timestamp=3200.0)
    # After snooze expires (+600s = 3600s), readings are accepted again
    assert not manager.record_reading(fatigue=0.9, stress=0.9, confidence=0.9, timestamp=3650.0)


def test_trial_anticipation_and_lapse_filtering(dummy_config):
    engine = CognitiveProbeEngine(dummy_config)

    trials = [
        # Anticipatory response (< 100ms)
        TrialResult("RED", "RED", "RED", reaction_time_ms=50.0, correct=True, is_anticipation=True, is_lapse=False),
        # Normal response
        TrialResult("GREEN", "GREEN", "GREEN", reaction_time_ms=280.0, correct=True, is_anticipation=False, is_lapse=False),
        # Attentional lapse (> 500ms)
        TrialResult("BLUE", "BLUE", "BLUE", reaction_time_ms=650.0, correct=True, is_anticipation=False, is_lapse=True),
    ]

    session_res, reading = engine.evaluate_session(ProbeType.PVT, trials, timestamp=100.0)

    # Anticipation should be excluded from valid RT list
    assert 50.0 not in session_res.valid_rt_list
    assert len(session_res.valid_rt_list) == 2
    assert session_res.anticipation_rate == pytest.approx(1.0 / 3.0)
    assert session_res.lapse_rate == pytest.approx(1.0 / 3.0)
    assert reading.source == ModalitySource.COGNITIVE_PROBE


def test_cognitive_fatigue_index_sensitivity(dummy_config):
    engine = CognitiveProbeEngine(dummy_config)
    engine.calibrate_baseline(pvt_median_rt=250.0, stroop_median_rt=500.0, lapse_rate=0.0)

    # Scenario A: Alert session (Fast RT, 0 lapses, 100% accuracy)
    alert_trials = [
        TrialResult("RED", "RED", "RED", 240.0, True, False, False),
        TrialResult("RED", "RED", "RED", 250.0, True, False, False),
        TrialResult("RED", "RED", "RED", 245.0, True, False, False),
    ]
    _, reading_alert = engine.evaluate_session(ProbeType.PVT, alert_trials)
    assert reading_alert.fatigue < 0.15

    # Scenario B: Severely fatigued session (+80% RT slowing, lapses, errors)
    exhausted_trials = [
        TrialResult("RED", "RED", "RED", 480.0, True, False, False),
        TrialResult("RED", "GREEN", "RED", 550.0, False, False, True),
        TrialResult("RED", "RED", "RED", 600.0, True, False, True),
    ]
    _, reading_fatigued = engine.evaluate_session(ProbeType.PVT, exhausted_trials)
    assert reading_fatigued.fatigue > 0.50
    assert reading_fatigued.valence < 0.0
