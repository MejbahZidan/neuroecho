"""Late fusion engine combining vision, audio, and cognitive probes into Russell's Circumplex.

Applies:
- Dynamic confidence-weighted fusion with missing-modality tolerance.
- Exponential Moving Average (EMA) for continuous scalar smoothing.
- Hysteresis state machine to eliminate discrete label flickering.
- Transparent explainability attribution.
"""

from collections import deque
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from core.schemas import AffectLabel, FusedState, ModalityReading, ModalitySource

logger = logging.getLogger(__name__)


class EmotionFusionEngine:
    """Fuses asynchronous multi-modal streams into a temporally stable FusedState."""

    DEFAULT_TTLS = {
        ModalitySource.VISION: 2.5,
        ModalitySource.AUDIO: 10.0,
        ModalitySource.COGNITIVE_PROBE: 120.0,
        ModalitySource.SELF_REPORT: 600.0,
    }

    PRIOR_WEIGHTS = {
        ModalitySource.VISION: 1.0,
        ModalitySource.AUDIO: 0.85,
        ModalitySource.COGNITIVE_PROBE: 1.2,
        ModalitySource.SELF_REPORT: 2.0,
    }

    def __init__(self, config: dict):
        self.config = config
        f_cfg = config.get("fusion", {})
        self.ema_alpha = f_cfg.get("smoothing_ema_alpha", 0.3)
        self.hysteresis_persistence_s = f_cfg.get("hysteresis_persistence_seconds", 3.0)

        self._latest_readings: Dict[ModalitySource, ModalityReading] = {}

        self._smoothed_valence: float = 0.0
        self._smoothed_arousal: float = 0.5
        self._smoothed_fatigue: float = 0.0
        self._smoothed_stress: float = 0.0
        self._has_prior: bool = False

        self._current_label: AffectLabel = AffectLabel.NEUTRAL
        self._candidate_label: AffectLabel = AffectLabel.NEUTRAL
        self._candidate_since: float = 0.0

        self._self_report_file = os.path.join("data", "self_reports.jsonl")

    def register_reading(self, reading: ModalityReading) -> None:
        self._latest_readings[reading.source] = reading

    def record_self_report(self, score: int, note: str = "", timestamp: Optional[float] = None) -> FusedState:
        now = timestamp if timestamp is not None else time.time()
        norm_score = (score - 3) / 2.0
        reading = ModalityReading(
            source=ModalitySource.SELF_REPORT,
            timestamp=now,
            valence=float(norm_score),
            arousal=float(0.5 + 0.3 * norm_score),
            fatigue=float(max(0.0, -norm_score)),
            stress=float(max(0.0, -norm_score * 0.5)),
            confidence=1.0,
            quality=1.0,
            features={"self_report_score": score, "note": note}
        )
        self.register_reading(reading)

        if self.config.get("app", {}).get("allow_local_telemetry", False):
            os.makedirs(os.path.dirname(self._self_report_file), exist_ok=True)
            with open(self._self_report_file, "a", encoding="utf-8") as f:
                record = {"timestamp": now, "score": score, "note": note}
                f.write(json.dumps(record) + "\n")

        return self.fuse(timestamp=now)

    def _determine_label(self, valence: float, arousal: float, fatigue: float, stress: float) -> AffectLabel:
        if abs(valence) < 0.15 and abs(arousal - 0.5) < 0.15 and fatigue < 0.30 and stress < 0.30:
            return AffectLabel.NEUTRAL

        if stress >= 0.55:
            return AffectLabel.TENSE_STRESSED
        if fatigue >= 0.55:
            return AffectLabel.LOW_ENERGY_FATIGUED

        if arousal >= 0.5:
            return AffectLabel.ENERGIZED_FOCUSED if valence >= 0.0 else AffectLabel.TENSE_STRESSED
        else:
            return AffectLabel.CALM if valence >= 0.0 else AffectLabel.LOW_ENERGY_FATIGUED

    def fuse(self, timestamp: Optional[float] = None) -> FusedState:
        now = timestamp if timestamp is not None else time.time()

        active_readings: List[Tuple[ModalityReading, float]] = []
        for source, reading in list(self._latest_readings.items()):
            ttl = self.DEFAULT_TTLS.get(source, 5.0)
            if (now - reading.timestamp) <= ttl and reading.confidence > 0.01:
                weight = reading.confidence * reading.quality * self.PRIOR_WEIGHTS.get(source, 1.0)
                if weight > 0.001:
                    active_readings.append((reading, weight))

        if not active_readings:
            return FusedState(
                valence=0.0,
                arousal=0.5,
                fatigue=0.0,
                stress=0.0,
                label=AffectLabel.NEUTRAL,
                confidence=0.0,
                contributing_modalities=[],
                timestamp=now,
                explanation={"none": 1.0}
            )

        total_weight = sum(w for _, w in active_readings)
        raw_v = sum(r.valence * w for r, w in active_readings) / total_weight
        raw_a = sum(r.arousal * w for r, w in active_readings) / total_weight
        raw_f = sum(r.fatigue * w for r, w in active_readings) / total_weight
        raw_s = sum(r.stress * w for r, w in active_readings) / total_weight

        base_conf = sum(r.confidence * w for r, w in active_readings) / total_weight
        synergy_bonus = min(0.15, 0.05 * (len(active_readings) - 1))
        fused_conf = float(min(1.0, base_conf + synergy_bonus))

        if not self._has_prior:
            self._smoothed_valence = raw_v
            self._smoothed_arousal = raw_a
            self._smoothed_fatigue = raw_f
            self._smoothed_stress = raw_s
            self._has_prior = True
        else:
            a = self.ema_alpha
            self._smoothed_valence = a * raw_v + (1.0 - a) * self._smoothed_valence
            self._smoothed_arousal = a * raw_a + (1.0 - a) * self._smoothed_arousal
            self._smoothed_fatigue = a * raw_f + (1.0 - a) * self._smoothed_fatigue
            self._smoothed_stress = a * raw_s + (1.0 - a) * self._smoothed_stress

        tentative_label = self._determine_label(
            self._smoothed_valence, self._smoothed_arousal,
            self._smoothed_fatigue, self._smoothed_stress
        )

        if tentative_label != self._current_label:
            if tentative_label != self._candidate_label:
                self._candidate_label = tentative_label
                self._candidate_since = now
            else:
                if (now - self._candidate_since) >= self.hysteresis_persistence_s:
                    self._current_label = self._candidate_label
        else:
            self._candidate_label = self._current_label
            self._candidate_since = now

        explanation = {
            r.source.value: round(float(w / total_weight), 3)
            for r, w in active_readings
        }

        return FusedState(
            valence=float(self._smoothed_valence),
            arousal=float(self._smoothed_arousal),
            fatigue=float(self._smoothed_fatigue),
            stress=float(self._smoothed_stress),
            label=self._current_label,
            confidence=fused_conf,
            contributing_modalities=[r.source for r, _ in active_readings],
            timestamp=now,
            explanation=explanation
        )
