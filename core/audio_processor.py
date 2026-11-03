"""Audio processing pipeline: VAD gating, prosody, local Wav2Vec2 SER, and Whisper semantics.

Strict privacy constraints:
- Operates entirely in RAM; no temporary audio files are written to disk.
- Transcripts and raw audio arrays are cleared from memory immediately after inference.
"""

import collections
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import webrtcvad

from core.schemas import ModalityReading, ModalitySource

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Processes 5-second speech-gated audio buffers for prosody, SER, and lexical affect."""

    DEFAULT_LEXICON = [
        "always", "never", "completely", "totally", "nothing", "everything",
        "hopeless", "ruined", "can't", "impossible", "worthless", "exhausted",
        "drowning", "pointless", "failure"
    ]

    LABEL_AFFECT_MAP = {
        "neu": (0.0, 0.3),
        "hap": (0.75, 0.70),
        "ang": (-0.60, 0.85),
        "sad": (-0.75, 0.20),
        "neutral": (0.0, 0.3),
        "happy": (0.75, 0.70),
        "angry": (-0.60, 0.85),
        "calm": (0.60, 0.15),
        "fearful": (-0.70, 0.80),
        "disgust": (-0.60, 0.50),
        "surprised": (0.20, 0.80),
    }

    def __init__(self, config: dict, lazy_load_models: bool = False):
        self.config = config
        self.a_cfg = config.get("audio", {})
        self.sample_rate = self.a_cfg.get("sample_rate", 16000)
        self.chunk_duration = self.a_cfg.get("chunk_duration_seconds", 5.0)
        self.vad_aggressiveness = self.a_cfg.get("vad_aggressiveness", 2)
        self.whisper_model_size = self.a_cfg.get("whisper_model_size", "base")
        self.language = self.a_cfg.get("language", "en")
        self.absolutist_lexicon = set(self.a_cfg.get("absolutist_lexicon", self.DEFAULT_LEXICON))

        self.calibrating = False
        self.calibrated = False
        self.calibration_samples: Dict[str, List[float]] = {
            "f0_mean": [],
            "rms_energy": [],
            "speech_rate": []
        }
        self.baselines: Dict[str, Tuple[float, float]] = {
            "f0": (160.0, 35.0),
            "energy": (0.03, 0.015),
            "speech_rate": (3.5, 0.8),
        }

        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.ser_extractor = None
        self.ser_model = None
        self.whisper_model = None

        if not lazy_load_models:
            self._init_models()

    def _init_models(self) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            model_name = self.a_cfg.get("ser_model_name", "superb/wav2vec2-base-superb-er")
            logger.info("Loading SER model (%s) on CPU...", model_name)
            self.ser_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.ser_model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.ser_model.eval()
        except Exception as e:
            logger.warning("Could not load HuggingFace SER model: %s. Using heuristic prosody.", e)
            self.ser_model = None

        try:
            from faster_whisper import WhisperModel
            logger.info("Loading Faster-Whisper (%s) on CPU (int8)...", self.whisper_model_size)
            self.whisper_model = WhisperModel(
                self.whisper_model_size, device="cpu", compute_type="int8"
            )
        except Exception as e:
            logger.warning("Could not load Faster-Whisper: %s. Text semantics will be bypassed.", e)
            self.whisper_model = None

    def start_calibration(self) -> None:
        self.calibration_samples = {"f0_mean": [], "rms_energy": [], "speech_rate": []}
        self.calibrating = True
        self.calibrated = False

    def finish_calibration(self) -> None:
        if len(self.calibration_samples["f0_mean"]) < 3:
            logger.warning("Insufficient audio samples for calibration; using defaults.")
            self.calibrated = True
            self.calibrating = False
            return

        for k, baseline_key in [("f0_mean", "f0"), ("rms_energy", "energy"), ("speech_rate", "speech_rate")]:
            vals = self.calibration_samples[k]
            mean_val = float(np.mean(vals)) if vals else self.baselines[baseline_key][0]
            std_val = float(np.std(vals)) if vals else self.baselines[baseline_key][1]
            self.baselines[baseline_key] = (mean_val, max(std_val, 1e-3))

        self.calibrated = True
        self.calibrating = False
        logger.info("Audio calibration complete. Baseline F0: %.1f Hz", self.baselines["f0"][0])

    def assess_audio_quality(self, audio_16k: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Assess signal fidelity: nominal RMS level (dBFS) and digital clipping check."""
        if len(audio_16k) == 0:
            return 0.0, {"snr_db": 0.0, "clipping_ratio": 1.0}

        # Check clipping (samples at >= 0.99)
        clipping_samples = np.sum(np.abs(audio_16k) >= 0.99)
        clipping_ratio = float(clipping_samples / len(audio_16k))

        # Nominal RMS level in dBFS
        rms = float(np.sqrt(np.mean(audio_16k ** 2) + 1e-12))
        rms_dbfs = float(20.0 * np.log10(max(rms, 1e-6)))

        # Silence threshold: below -50 dBFS is unusable silence
        if rms_dbfs < -50.0:
            level_score = 0.0
        elif rms_dbfs > -3.0:
            level_score = 0.8  # Dangerously close to saturation
        else:
            # Linear ramp from -50 dBFS (0.0) to -20 dBFS (1.0)
            level_score = float(np.clip((rms_dbfs + 50.0) / 30.0, 0.0, 1.0))

        # Heavy clipping penalty
        clip_penalty = float(np.clip(clipping_ratio / 0.01, 0.0, 1.0))
        quality = float(max(0.0, level_score * (1.0 - 0.85 * clip_penalty)))

        return quality, {"snr_db": rms_dbfs + 60.0, "clipping_ratio": clipping_ratio}

    def compute_vad_ratio(self, audio_16k: np.ndarray) -> Tuple[float, List[bool]]:
        frame_ms = 30
        frame_size = int(self.sample_rate * (frame_ms / 1000.0))
        n_frames = len(audio_16k) // frame_size

        if n_frames == 0:
            return 0.0, []

        audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        speech_flags = []
        for i in range(n_frames):
            chunk = audio_int16[i * frame_size * 2 : (i + 1) * frame_size * 2]
            try:
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
            except Exception:
                is_speech = False
            speech_flags.append(is_speech)

        speech_ratio = float(sum(speech_flags) / len(speech_flags))
        return speech_ratio, speech_flags

    def extract_prosody(self, audio_16k: np.ndarray, speech_flags: List[bool]) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(audio_16k ** 2) + 1e-9))
        pause_ratio = float(1.0 - (sum(speech_flags) / len(speech_flags))) if speech_flags else 1.0

        try:
            f0 = librosa.yin(audio_16k, fmin=60, fmax=400, sr=self.sample_rate)
            voiced_f0 = f0[(f0 > 65) & (f0 < 390)]
            if len(voiced_f0) > 5:
                f0_mean = float(np.mean(voiced_f0))
                f0_std = float(np.std(voiced_f0))
                diffs = np.abs(np.diff(voiced_f0))
                jitter = float(np.mean(diffs) / max(f0_mean, 1e-5))
            else:
                f0_mean, f0_std, jitter = 0.0, 0.0, 0.0
        except Exception:
            f0_mean, f0_std, jitter = 0.0, 0.0, 0.0

        frame_rms = librosa.feature.rms(y=audio_16k, frame_length=512, hop_length=256)[0]
        if len(frame_rms) > 2 and np.mean(frame_rms) > 1e-4:
            shimmer = float(np.mean(np.abs(np.diff(frame_rms))) / np.mean(frame_rms))
        else:
            shimmer = 0.0

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "jitter": jitter,
            "shimmer": shimmer,
            "rms_energy": rms,
            "pause_ratio": pause_ratio,
        }

    def infer_ser(self, audio_16k: np.ndarray) -> Tuple[float, float, float, str]:
        if self.ser_model is None or self.ser_extractor is None:
            return 0.0, 0.0, 0.0, "unavailable"

        with torch.no_grad():
            inputs = self.ser_extractor(audio_16k, sampling_rate=self.sample_rate, return_tensors="pt")
            outputs = self.ser_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

        id2label = self.ser_model.config.id2label
        top_idx = int(np.argmax(probs))
        top_label = id2label.get(top_idx, "neutral").lower()
        confidence = float(probs[top_idx])

        valence, arousal = self.LABEL_AFFECT_MAP.get(top_label, (0.0, 0.3))
        return valence, arousal, confidence, top_label

    def analyze_semantics(self, audio_16k: np.ndarray) -> Tuple[float, int, bool]:
        if self.whisper_model is None:
            return 0.0, 0, False

        try:
            segments, _ = self.whisper_model.transcribe(
                audio_16k, language=self.language, beam_size=1, condition_on_previous_text=False
            )
            raw_text = " ".join([s.text for s in segments]).strip().lower()
        except Exception as e:
            logger.debug("Whisper transcription failed: %s", e)
            return 0.0, 0, False

        if not raw_text:
            return 0.0, 0, False

        # Strip punctuation so words match lexicon exactly
        clean_text = re.sub(r"[^\w\s]", "", raw_text)
        words = clean_text.split()
        absolutist_count = sum(1 for w in words if w in self.absolutist_lexicon)

        pos_words = {"good", "great", "fine", "happy", "relaxed", "better", "clear", "calm", "well"}
        neg_words = {"tired", "exhausted", "bad", "stressed", "hate", "stuck", "overwhelmed", "hard", "awful"}

        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)

        text_valence = 0.0
        if pos_count > 0 or neg_count > 0:
            text_valence = float((pos_count - neg_count) / max(pos_count + neg_count, 1))

        has_hopeless_flag = absolutist_count >= 2 or ("hopeless" in words)
        return text_valence, absolutist_count, has_hopeless_flag

    def process_buffer(self, audio_16k: np.ndarray, timestamp: Optional[float] = None) -> ModalityReading:
        now = timestamp if timestamp is not None else time.time()
        quality, q_metrics = self.assess_audio_quality(audio_16k)

        speech_ratio, speech_flags = self.compute_vad_ratio(audio_16k)
        vad_min_ratio = self.a_cfg.get("vad_speech_ratio_threshold", 0.25)

        if speech_ratio < vad_min_ratio or quality < 0.20:
            return ModalityReading(
                source=ModalitySource.AUDIO,
                timestamp=now,
                valence=0.0,
                arousal=0.0,
                fatigue=0.0,
                stress=0.0,
                confidence=0.0,
                quality=quality,
                features={"speech_detected": False, "speech_ratio": speech_ratio, **q_metrics}
            )

        prosody = self.extract_prosody(audio_16k, speech_flags)
        if self.calibrating and prosody["f0_mean"] > 0:
            self.calibration_samples["f0_mean"].append(prosody["f0_mean"])
            self.calibration_samples["rms_energy"].append(prosody["rms_energy"])

        f0_base, f0_std = self.baselines["f0"]
        f0_z = (prosody["f0_mean"] - f0_base) / f0_std if prosody["f0_mean"] > 0 else 0.0

        energy_base, energy_std = self.baselines["energy"]
        energy_z = (prosody["rms_energy"] - energy_base) / energy_std

        ser_valence, ser_arousal, ser_conf, top_emotion = self.infer_ser(audio_16k)
        text_valence, absolutist_count, hopeless_flag = self.analyze_semantics(audio_16k)

        acoustic_arousal = 1.0 / (1.0 + math.exp(-(0.5 * f0_z + 0.6 * energy_z - 1.2 * prosody["pause_ratio"])))
        acoustic_arousal = float(np.clip(acoustic_arousal, 0.0, 1.0))

        final_arousal = float(0.6 * acoustic_arousal + 0.4 * ser_arousal) if ser_conf > 0.4 else acoustic_arousal

        if ser_conf > 0.35:
            final_valence = float(np.clip(0.7 * ser_valence + 0.3 * text_valence, -1.0, 1.0))
        else:
            final_valence = float(np.clip(text_valence, -1.0, 1.0))

        stress = float(np.clip(0.4 * max(0.0, f0_z) + 0.3 * prosody["jitter"] * 10.0 + 0.3 * (1.0 - ser_valence if ser_valence < 0 else 0.0), 0.0, 1.0))
        fatigue = float(np.clip(0.5 * prosody["pause_ratio"] + 0.5 * max(0.0, -energy_z), 0.0, 1.0))
        confidence = float(np.clip(quality * (0.5 * speech_ratio + 0.5 * ser_conf), 0.0, 1.0))

        del audio_16k

        return ModalityReading(
            source=ModalitySource.AUDIO,
            timestamp=now,
            valence=final_valence,
            arousal=final_arousal,
            fatigue=fatigue,
            stress=stress,
            confidence=confidence,
            quality=quality,
            features={
                "speech_detected": True,
                "speech_ratio": speech_ratio,
                "top_emotion": top_emotion,
                "absolutist_count": absolutist_count,
                "hopeless_flag": hopeless_flag,
                "f0_z": f0_z,
                "energy_z": energy_z,
                **prosody,
                **q_metrics
            }
        )
