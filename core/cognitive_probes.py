"""Cognitive probes module: Stroop Effect, Brief PVT, trigger rules, and fatigue index calculation.

Scientific consideration:
Short interactive probes (15-45 seconds) introduce noise from motor variability.
Their output (ModalityReading) acts as an active verification signal with bounded
confidence (ceiling 0.75), rather than an infallible ground-truth arbiter.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import random
import time
from typing import Deque, Dict, List, Optional, Tuple

from core.schemas import ModalityReading, ModalitySource

logger = logging.getLogger(__name__)


class ProbeType(str, Enum):
    STROOP = "stroop"
    PVT = "pvt"


@dataclass(frozen=True)
class TrialResult:
    stimulus: str
    response: str
    target: str
    reaction_time_ms: float
    correct: bool
    is_anticipation: bool   # RT < 100ms
    is_lapse: bool          # RT > 500ms or timeout


@dataclass
class ProbeSessionResult:
    probe_type: ProbeType
    timestamp: float
    trials: List[TrialResult]
    valid_rt_list: List[float]
    median_rt_ms: float
    mean_rt_ms: float
    accuracy: float
    lapse_rate: float
    anticipation_rate: float
    cognitive_fatigue_index: float  # [0.0, 1.0]


class ProbeTriggerManager:
    """Enforces cooldowns, frequency limits, and sustained-threshold trigger criteria."""

    def __init__(self, config: dict):
        self.config = config
        p_cfg = config.get("probes", {})
        self.trigger_threshold = p_cfg.get("trigger_threshold", 0.60)
        self.consecutive_required = p_cfg.get("consecutive_triggers", 3)
        self.cooldown_seconds = p_cfg.get("cooldown_minutes", 30) * 60.0
        self.probes_per_hour = p_cfg.get("probes_per_hour", 2)

        self._consecutive_timestamps: Deque[float] = deque()
        self._completed_timestamps: Deque[float] = deque()
        self._snooze_until: float = 0.0

    def record_reading(self, fatigue: float, stress: float, confidence: float, timestamp: Optional[float] = None) -> bool:
        now = timestamp if timestamp is not None else time.time()

        if now < self._snooze_until:
            return False

        if self._completed_timestamps:
            last_completed = self._completed_timestamps[-1]
            if (now - last_completed) < self.cooldown_seconds:
                return False

        one_hour_ago = now - 3600.0
        while self._completed_timestamps and self._completed_timestamps[0] < one_hour_ago:
            self._completed_timestamps.popleft()
        if len(self._completed_timestamps) >= self.probes_per_hour:
            return False

        is_elevated = (confidence >= 0.50) and ((fatigue >= self.trigger_threshold) or (stress >= self.trigger_threshold))

        if is_elevated:
            self._consecutive_timestamps.append(now)
            while self._consecutive_timestamps and (now - self._consecutive_timestamps[0] > 180.0):
                self._consecutive_timestamps.popleft()

            if len(self._consecutive_timestamps) >= self.consecutive_required:
                time_span = self._consecutive_timestamps[-1] - self._consecutive_timestamps[0]
                if time_span >= 60.0:
                    return True
        else:
            if self._consecutive_timestamps:
                self._consecutive_timestamps.popleft()

        return False

    def snooze(self, minutes: int = 10, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        self._snooze_until = now + (minutes * 60.0)
        self._consecutive_timestamps.clear()
        logger.info("Cognitive probe snoozed for %d minutes.", minutes)

    def skip(self, timestamp: Optional[float] = None) -> None:
        self.snooze(minutes=15, timestamp=timestamp)

    def mark_completed(self, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        self._completed_timestamps.append(now)
        self._consecutive_timestamps.clear()


class CognitiveProbeEngine:
    """Generates trials and calculates cognitive strain/fatigue relative to user baseline."""

    STROOP_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]

    def __init__(self, config: dict):
        self.config = config
        self.trigger_manager = ProbeTriggerManager(config)

        self.baseline_pvt_median_rt = 280.0
        self.baseline_stroop_median_rt = 600.0
        self.baseline_lapse_rate = 0.05
        self.is_calibrated = False

    def calibrate_baseline(self, pvt_median_rt: float, stroop_median_rt: float, lapse_rate: float = 0.05) -> None:
        self.baseline_pvt_median_rt = max(180.0, pvt_median_rt)
        self.baseline_stroop_median_rt = max(350.0, stroop_median_rt)
        self.baseline_lapse_rate = max(0.01, lapse_rate)
        self.is_calibrated = True
        logger.info("Cognitive probe baseline calibrated.")

    def generate_stroop_trials(self, n_trials: int = 12) -> List[Dict[str, str]]:
        trials = []
        for _ in range(n_trials):
            word = random.choice(self.STROOP_COLORS)
            if random.random() < 0.5:
                ink_color = word
            else:
                ink_color = random.choice([c for c in self.STROOP_COLORS if c != word])
            trials.append({"word": word, "color": ink_color, "target": ink_color})
        return trials

    def generate_pvt_delays(self, n_trials: int = 8, min_delay_s: float = 1.5, max_delay_s: float = 4.0) -> List[float]:
        return [random.uniform(min_delay_s, max_delay_s) for _ in range(n_trials)]

    def evaluate_session(
        self,
        probe_type: ProbeType,
        raw_trials: List[TrialResult],
        timestamp: Optional[float] = None
    ) -> Tuple[ProbeSessionResult, ModalityReading]:
        now = timestamp if timestamp is not None else time.time()
        self.trigger_manager.mark_completed(now)

        if not raw_trials:
            empty_res = ProbeSessionResult(
                probe_type=probe_type, timestamp=now, trials=[], valid_rt_list=[],
                median_rt_ms=0.0, mean_rt_ms=0.0, accuracy=0.0, lapse_rate=0.0,
                anticipation_rate=0.0, cognitive_fatigue_index=0.0
            )
            empty_reading = ModalityReading(
                source=ModalitySource.COGNITIVE_PROBE, timestamp=now,
                valence=0.0, arousal=0.0, fatigue=0.0, stress=0.0, confidence=0.0, quality=0.0
            )
            return empty_res, empty_reading

        valid_rts = [t.reaction_time_ms for t in raw_trials if not t.is_anticipation and t.reaction_time_ms > 0]
        correct_count = sum(1 for t in raw_trials if t.correct)
        lapse_count = sum(1 for t in raw_trials if t.is_lapse)
        anticipation_count = sum(1 for t in raw_trials if t.is_anticipation)
        total_trials = len(raw_trials)

        accuracy = float(correct_count / total_trials)
        lapse_rate = float(lapse_count / total_trials)
        anticipation_rate = float(anticipation_count / total_trials)

        median_rt = float(sorted(valid_rts)[len(valid_rts) // 2]) if valid_rts else 800.0
        mean_rt = float(sum(valid_rts) / len(valid_rts)) if valid_rts else 800.0

        base_rt = self.baseline_stroop_median_rt if probe_type == ProbeType.STROOP else self.baseline_pvt_median_rt
        rt_slowing = max(0.0, (median_rt - base_rt) / base_rt)
        lapse_penalty = max(0.0, lapse_rate - self.baseline_lapse_rate)
        accuracy_penalty = max(0.0, 1.0 - accuracy)

        cfi = float(min(1.0, 0.5 * min(1.0, rt_slowing * 2.5) + 0.3 * (lapse_penalty * 2.0) + 0.2 * accuracy_penalty))

        session_result = ProbeSessionResult(
            probe_type=probe_type,
            timestamp=now,
            trials=raw_trials,
            valid_rt_list=valid_rts,
            median_rt_ms=median_rt,
            mean_rt_ms=mean_rt,
            accuracy=accuracy,
            lapse_rate=lapse_rate,
            anticipation_rate=anticipation_rate,
            cognitive_fatigue_index=cfi
        )

        trial_ratio = min(1.0, len(valid_rts) / 8.0)
        confidence = float(0.75 * trial_ratio * (1.0 - anticipation_rate))

        valence = float(-0.6 * cfi - 0.4 * accuracy_penalty)
        arousal = float(max(0.1, 1.0 - cfi))
        stress = float(min(1.0, 0.7 * accuracy_penalty + 0.3 * cfi))

        reading = ModalityReading(
            source=ModalitySource.COGNITIVE_PROBE,
            timestamp=now,
            valence=valence,
            arousal=arousal,
            fatigue=cfi,
            stress=stress,
            confidence=confidence,
            quality=1.0 - anticipation_rate,
            features={
                "probe_type": probe_type.value,
                "median_rt_ms": median_rt,
                "lapse_rate": lapse_rate,
                "accuracy": accuracy,
                "cognitive_fatigue_index": cfi,
                "rt_slowing_ratio": rt_slowing,
            }
        )

        return session_result, reading
