"""Unit tests for the Emotion Fusion Engine (missing modalities, EMA, hysteresis, explanation)."""

import pytest

from core.fusion_engine import EmotionFusionEngine
from core.schemas import AffectLabel, ModalityReading, ModalitySource


@pytest.fixture
def dummy_config():
    return {
        "app": {"allow_local_telemetry": False},
        "fusion": {
            "smoothing_ema_alpha": 0.5,
            "hysteresis_persistence_seconds": 2.0,
        }
    }


def test_missing_modality_handling(dummy_config):
    engine = EmotionFusionEngine(dummy_config)

    # 1. No readings at all -> Neutral with 0 confidence
    fused_empty = engine.fuse(timestamp=100.0)
    assert fused_empty.label == AffectLabel.NEUTRAL
    assert fused_empty.confidence == 0.0
    assert fused_empty.contributing_modalities == []

    # 2. Vision only available -> Seamlessly fuses without error
    reading_v = ModalityReading(
        source=ModalitySource.VISION,
        timestamp=100.0,
        valence=0.6,
        arousal=0.7,
        fatigue=0.1,
        stress=0.1,
        confidence=0.8,
        quality=0.9
    )
    engine.register_reading(reading_v)
    fused_v = engine.fuse(timestamp=100.0)

    assert fused_v.contributing_modalities == [ModalitySource.VISION]
    assert fused_v.confidence > 0.7
    assert "vision" in fused_v.explanation


def test_confidence_weighted_override(dummy_config):
    engine = EmotionFusionEngine(dummy_config)

    # High-confidence vision (calm: positive valence)
    vision_reading = ModalityReading(
        source=ModalitySource.VISION, timestamp=100.0,
        valence=0.8, arousal=0.3, fatigue=0.1, stress=0.1,
        confidence=0.9, quality=1.0
    )
    # Very low-confidence audio (discordant negative valence)
    audio_reading = ModalityReading(
        source=ModalitySource.AUDIO, timestamp=100.0,
        valence=-0.9, arousal=0.9, fatigue=0.8, stress=0.8,
        confidence=0.1, quality=0.3
    )

    engine.register_reading(vision_reading)
    engine.register_reading(audio_reading)
    fused = engine.fuse(timestamp=100.0)

    # Output valence must stay strongly positive because vision weight >> audio weight
    assert fused.valence > 0.4
    assert fused.explanation["vision"] > fused.explanation["audio"]


def test_ema_smoothing(dummy_config):
    # Alpha = 0.5
    engine = EmotionFusionEngine(dummy_config)

    # Reading 1: Valence = 0.0
    r1 = ModalityReading(
        source=ModalitySource.VISION, timestamp=10.0,
        valence=0.0, arousal=0.5, fatigue=0.0, stress=0.0,
        confidence=0.8, quality=1.0
    )
    engine.register_reading(r1)
    f1 = engine.fuse(timestamp=10.0)
    assert f1.valence == pytest.approx(0.0)

    # Reading 2: Sudden spike to 1.0 at t=11s
    r2 = ModalityReading(
        source=ModalitySource.VISION, timestamp=11.0,
        valence=1.0, arousal=0.5, fatigue=0.0, stress=0.0,
        confidence=0.8, quality=1.0
    )
    engine.register_reading(r2)
    f2 = engine.fuse(timestamp=11.0)

    # With alpha=0.5, EMA value must be exactly 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert f2.valence == pytest.approx(0.5)


def test_hysteresis_prevents_flickering(dummy_config):
    # Hysteresis persistence is set to 2.0s
    engine = EmotionFusionEngine(dummy_config)

    # Establish baseline CALM state (arousal=0.3, valence=0.6)
    r_calm = ModalityReading(
        source=ModalitySource.VISION, timestamp=100.0,
        valence=0.6, arousal=0.3, fatigue=0.0, stress=0.0,
        confidence=0.9, quality=1.0
    )
    engine.register_reading(r_calm)
    f0 = engine.fuse(timestamp=100.0)
    # Force initial label to CALM after duration
    engine._current_label = AffectLabel.CALM
    engine._candidate_label = AffectLabel.CALM

    # Sudden transient stress reading at t=101.0s (1.0s later < 2.0s threshold)
    r_stress = ModalityReading(
        source=ModalitySource.VISION, timestamp=101.0,
        valence=-0.8, arousal=0.9, fatigue=0.0, stress=0.9,
        confidence=0.9, quality=1.0
    )
    engine.register_reading(r_stress)
    f_transient = engine.fuse(timestamp=101.0)
    # Label should STILL be CALM because 1 second < 2 seconds required
    assert f_transient.label == AffectLabel.CALM

    # After 2.5s of persistent stress reading (t=103.5s), label must finally transition
    r_stress_cont = ModalityReading(
        source=ModalitySource.VISION, timestamp=103.5,
        valence=-0.8, arousal=0.9, fatigue=0.0, stress=0.9,
        confidence=0.9, quality=1.0
    )
    engine.register_reading(r_stress_cont)
    f_sustained = engine.fuse(timestamp=103.5)
    assert f_sustained.label == AffectLabel.TENSE_STRESSED
