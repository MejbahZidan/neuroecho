"""Unit tests for the Intervention Manager (posture streaks, Iso-principle music, safety escalation)."""

import pytest

from core.intervention_manager import (
    BreathingTechnique,
    InterventionManager,
    InterventionType,
    LocalMusicProvider,
    MusicTrack
)
from core.schemas import AffectLabel, FusedState, ModalitySource


@pytest.fixture
def dummy_config():
    return {
        "app": {"allow_local_telemetry": False},
        "interventions": {
            "posture_bad_streak_minutes": 15,
            "posture_reminder_interval_minutes": 30,
            "breathing_default": "4-4-4-4",
        }
    }


def test_posture_streak_and_cooldown(dummy_config):
    manager = InterventionManager(dummy_config)
    fused_neutral = FusedState(0.0, 0.5, 0.0, 0.0, AffectLabel.NEUTRAL, 0.8, [], 0.0)

    # 1. Slouch begins at t=100s -> No alert yet
    recs = manager.evaluate_interventions(fused_neutral, posture_score=0.40, timestamp=100.0)
    assert len(recs) == 0

    # 2. Slouch at t=500s (< 15 mins / 900s) -> No alert
    recs = manager.evaluate_interventions(fused_neutral, posture_score=0.40, timestamp=500.0)
    assert len(recs) == 0

    # 3. Slouch reaches 15 mins (t=1000s, span = 900s) -> Posture alert triggers!
    recs = manager.evaluate_interventions(fused_neutral, posture_score=0.40, timestamp=1000.0)
    assert len(recs) == 1
    assert recs[0].type == InterventionType.POSTURE_RESET

    # 4. Immediate subsequent check (t=1050s) within 30-min cooldown -> Suppressed
    recs_cooldown = manager.evaluate_interventions(fused_neutral, posture_score=0.40, timestamp=1050.0)
    assert len(recs_cooldown) == 0


def test_iso_principle_music_progression():
    provider = LocalMusicProvider()
    playlist = provider.get_iso_playlist(
        current_valence=-0.8, current_arousal=0.2,
        target_valence=0.6, target_arousal=0.5, num_tracks=4
    )

    assert len(playlist) == 4
    assert playlist[0].valence < 0.0
    assert playlist[-1].valence > 0.3
    assert playlist[-1].valence > playlist[0].valence


def test_safety_escalation_multi_day(dummy_config):
    manager = InterventionManager(dummy_config)
    fused_distress = FusedState(-0.7, 0.8, 0.8, 0.8, AffectLabel.TENSE_STRESSED, 0.8, [], 0.0)

    manager._distress_days = ["2026-10-30", "2026-10-31", "2026-11-01"]

    recs = manager.evaluate_interventions(fused_distress, posture_score=0.90, timestamp=1700000.0)
    assert len(recs) == 1
    assert recs[0].type == InterventionType.SAFETY_CHECKIN
    assert "professional" in recs[0].description
