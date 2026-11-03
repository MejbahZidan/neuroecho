from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModalitySource(str, Enum):
    VISION = "vision"
    AUDIO = "audio"
    COGNITIVE_PROBE = "cognitive_probe"
    SELF_REPORT = "self_report"


class AffectLabel(str, Enum):
    TENSE_STRESSED = "Tense / Stressed"
    LOW_ENERGY_FATIGUED = "Low-energy / Fatigued"
    CALM = "Calm"
    ENERGIZED_FOCUSED = "Energized / Focused"
    NEUTRAL = "Neutral"


@dataclass(frozen=True)
class ModalityReading:
    source: ModalitySource
    timestamp: float
    valence: float
    arousal: float
    fatigue: float
    stress: float
    confidence: float
    quality: float
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedState:
    valence: float
    arousal: float
    fatigue: float
    stress: float
    label: AffectLabel
    confidence: float
    contributing_modalities: List[ModalitySource]
    timestamp: float
    explanation: Dict[str, float] = field(default_factory=dict)
