"""Unit tests for audio processing algorithms (VAD gating, prosody, quality, lexicon parsing)."""

import numpy as np
import pytest

from core.audio_processor import AudioProcessor
from core.schemas import ModalitySource


@pytest.fixture
def dummy_config():
    return {
        "audio": {
            "sample_rate": 16000,
            "chunk_duration_seconds": 5.0,
            "vad_aggressiveness": 2,
            "vad_speech_ratio_threshold": 0.20,
            "whisper_model_size": "base",
            "language": "en",
            "absolutist_lexicon": ["always", "never", "completely", "hopeless"]
        }
    }


def test_audio_quality_metrics(dummy_config):
    # Initialize without heavy transformers for fast unit testing
    processor = AudioProcessor(dummy_config, lazy_load_models=True)

    # 1. Total silence
    silence = np.zeros(16000 * 2, dtype=np.float32)
    quality, metrics = processor.assess_audio_quality(silence)
    assert quality <= 0.5
    assert metrics["clipping_ratio"] == 0.0

    # 2. Clean 440Hz sine wave (high SNR, no clipping)
    t = np.linspace(0, 2.0, 16000 * 2, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    q_sine, m_sine = processor.assess_audio_quality(sine)
    assert q_sine > 0.5
    assert m_sine["clipping_ratio"] == 0.0

    # 3. Severely clipped audio
    clipped = np.full(16000 * 2, 1.0, dtype=np.float32)
    q_clip, m_clip = processor.assess_audio_quality(clipped)
    assert m_clip["clipping_ratio"] == 1.0
    assert q_clip < 0.3


def test_prosody_extraction(dummy_config):
    processor = AudioProcessor(dummy_config, lazy_load_models=True)

    # Synthesize 1 second of 200 Hz tone (within human vocal pitch range)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    speech_flags = [True] * 33

    prosody = processor.extract_prosody(tone, speech_flags)
    assert prosody["f0_mean"] == pytest.approx(200.0, abs=10.0)
    assert prosody["rms_energy"] > 0.3
    assert prosody["pause_ratio"] == 0.0


def test_vad_silence_gating(dummy_config):
    processor = AudioProcessor(dummy_config, lazy_load_models=True)

    # 5 seconds of silence should return speech_detected=False and confidence=0.0
    silence = np.zeros(16000 * 5, dtype=np.float32)
    reading = processor.process_buffer(silence)

    assert reading.source == ModalitySource.AUDIO
    assert reading.confidence == 0.0
    assert not reading.features["speech_detected"]


def test_absolutist_lexicon_logic(dummy_config):
    processor = AudioProcessor(dummy_config, lazy_load_models=True)

    # Mock Whisper output with known absolutist markers
    class MockSegment:
        def __init__(self, text):
            self.text = text

    class MockWhisper:
        def transcribe(self, audio, **kwargs):
            return [MockSegment("This is always impossible and completely hopeless.")], None

    processor.whisper_model = MockWhisper()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    val, abs_count, hopeless_flag = processor.analyze_semantics(dummy_audio)

    assert abs_count == 3  # "always", "completely", "hopeless"
    assert hopeless_flag is True
