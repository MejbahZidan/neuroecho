"""Intervention manager: guided breathing, posture ergonomic alerts, Iso-principle music, and safety rules."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.schemas import AffectLabel, FusedState

logger = logging.getLogger(__name__)


class BreathingTechnique(str, Enum):
    BOX_4_4_4_4 = "4-4-4-4"
    RELAXING_4_7_8 = "4-7-8"


class InterventionType(str, Enum):
    BREATHING = "breathing"
    POSTURE_RESET = "posture_reset"
    MUSIC_PLAYLIST = "music_playlist"
    SAFETY_CHECKIN = "safety_checkin"


@dataclass(frozen=True)
class MusicTrack:
    title: str
    artist: str
    valence: float     # [-1.0, 1.0]
    arousal: float     # [0.0, 1.0]
    uri_or_path: str
    duration_s: int = 180


@dataclass(frozen=True)
class InterventionRecommendation:
    type: InterventionType
    title: str
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class MusicProvider(ABC):
    @abstractmethod
    def get_iso_playlist(
        self,
        current_valence: float,
        current_arousal: float,
        target_valence: float = 0.5,
        target_arousal: float = 0.4,
        num_tracks: int = 4
    ) -> List[MusicTrack]:
        pass


class LocalMusicProvider(MusicProvider):
    DEFAULT_CATALOG = [
        MusicTrack("Drifting Embers", "Ambient Minimal", -0.6, 0.2, "local://tracks/somber_1.mp3"),
        MusicTrack("Rain Over Cedar", "Solo Piano", -0.4, 0.25, "local://tracks/somber_2.mp3"),
        MusicTrack("Tension Wire", "Downtempo Strings", -0.5, 0.7, "local://tracks/tense_1.mp3"),
        MusicTrack("Still Horizon", "Lo-Fi Focus", 0.0, 0.35, "local://tracks/neutral_1.mp3"),
        MusicTrack("Warm Dust", "Acoustic Fingerstyle", 0.1, 0.4, "local://tracks/neutral_2.mp3"),
        MusicTrack("Morning Mist", "Neo-Classical", 0.45, 0.3, "local://tracks/calm_1.mp3"),
        MusicTrack("Clear Glade", "Ambient Pad", 0.6, 0.25, "local://tracks/calm_2.mp3"),
        MusicTrack("Sunlight On Stone", "Uptempo Indie", 0.7, 0.65, "local://tracks/energetic_1.mp3"),
        MusicTrack("Bright Ascent", "Electronic Beats", 0.8, 0.75, "local://tracks/energetic_2.mp3"),
    ]

    def __init__(self, catalog: Optional[List[MusicTrack]] = None):
        self.catalog = catalog if catalog is not None else self.DEFAULT_CATALOG

    def get_iso_playlist(
        self,
        current_valence: float,
        current_arousal: float,
        target_valence: float = 0.5,
        target_arousal: float = 0.4,
        num_tracks: int = 4
    ) -> List[MusicTrack]:
        playlist: List[MusicTrack] = []
        used_uris = set()

        for step in range(num_tracks):
            ratio = step / max(1, num_tracks - 1)
            step_v = current_valence + ratio * (target_valence - current_valence)
            step_a = current_arousal + ratio * (target_arousal - current_arousal)

            best_track = None
            min_dist = float("inf")
            for track in self.catalog:
                if track.uri_or_path in used_uris:
                    continue
                dist = (track.valence - step_v) ** 2 + (track.arousal - step_a) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_track = track

            if best_track is not None:
                playlist.append(best_track)
                used_uris.add(best_track.uri_or_path)

        return playlist


class InterventionManager:
    def __init__(self, config: dict, music_provider: Optional[MusicProvider] = None):
        self.config = config
        i_cfg = config.get("interventions", {})
        self.posture_streak_threshold_s = i_cfg.get("posture_bad_streak_minutes", 15) * 60.0
        self.posture_cooldown_s = i_cfg.get("posture_reminder_interval_minutes", 30) * 60.0
        self.breathing_default = BreathingTechnique(i_cfg.get("breathing_default", "4-4-4-4"))

        self.music_provider = music_provider or LocalMusicProvider()

        self._bad_posture_start_time: Optional[float] = None
        self._last_posture_reminder_time: float = -float("inf")

        self._distress_days_file = os.path.join("data", "distress_days.json")
        self._distress_days = self._load_distress_days()
        self._last_safety_checkin_time: float = -float("inf")

    def _load_distress_days(self) -> List[str]:
        if os.path.exists(self._distress_days_file):
            try:
                with open(self._distress_days_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _record_distress_day(self, day_str: str) -> None:
        if day_str not in self._distress_days:
            self._distress_days.append(day_str)
            self._distress_days = self._distress_days[-14:]
            if self.config.get("app", {}).get("allow_local_telemetry", False):
                os.makedirs(os.path.dirname(self._distress_days_file), exist_ok=True)
                with open(self._distress_days_file, "w", encoding="utf-8") as f:
                    json.dump(self._distress_days, f)

    def evaluate_interventions(
        self,
        fused_state: FusedState,
        posture_score: float,
        timestamp: Optional[float] = None
    ) -> List[InterventionRecommendation]:
        now = timestamp if timestamp is not None else time.time()
        recommendations: List[InterventionRecommendation] = []

        # 1. Posture Check
        if posture_score < 0.60:
            if self._bad_posture_start_time is None:
                self._bad_posture_start_time = now
            elif (now - self._bad_posture_start_time) >= self.posture_streak_threshold_s:
                if (now - self._last_posture_reminder_time) >= self.posture_cooldown_s:
                    recommendations.append(InterventionRecommendation(
                        type=InterventionType.POSTURE_RESET,
                        title="Posture Reset Reminder",
                        description="You've been leaning forward or slouched for over 15 minutes. Roll your shoulders back and level your chin.",
                        payload={"posture_score": round(posture_score, 2)},
                        timestamp=now
                    ))
                    self._last_posture_reminder_time = now
        else:
            self._bad_posture_start_time = None

        # 2. Multi-Day Safety Check-in
        if fused_state.confidence >= 0.60 and (fused_state.stress >= 0.70 or fused_state.fatigue >= 0.80):
            today_str = time.strftime("%Y-%m-%d", time.localtime(now))
            self._record_distress_day(today_str)

        if len(self._distress_days) >= 3 and (now - self._last_safety_checkin_time) >= 86400.0:
            recommendations.append(InterventionRecommendation(
                type=InterventionType.SAFETY_CHECKIN,
                title="Supportive Check-in",
                description="We've noticed you've felt tense or depleted across several recent days. Consider taking a dedicated break or speaking with someone you trust or a professional.",
                payload={"days_flagged": len(self._distress_days)},
                timestamp=now
            ))
            self._last_safety_checkin_time = now
            return recommendations

        # 3. Acute Stress -> Box Breathing
        if fused_state.label == AffectLabel.TENSE_STRESSED and fused_state.confidence >= 0.55:
            recommendations.append(InterventionRecommendation(
                type=InterventionType.BREATHING,
                title="Guided Box Breathing",
                description="A short 1-minute 4-4-4-4 breathing cycle can help calm autonomic tension.",
                payload={
                    "technique": self.breathing_default.value,
                    "cycle_seconds": [4, 4, 4, 4] if self.breathing_default == BreathingTechnique.BOX_4_4_4_4 else [4, 7, 8, 0]
                },
                timestamp=now
            ))

        # 4. Low Energy / Stress -> Iso-Principle Music
        if fused_state.label in (AffectLabel.TENSE_STRESSED, AffectLabel.LOW_ENERGY_FATIGUED) and fused_state.confidence >= 0.55:
            target_v = 0.50
            target_a = 0.35 if fused_state.label == AffectLabel.TENSE_STRESSED else 0.60
            playlist = self.music_provider.get_iso_playlist(
                current_valence=fused_state.valence,
                current_arousal=fused_state.arousal,
                target_valence=target_v,
                target_arousal=target_a,
                num_tracks=4
            )
            recommendations.append(InterventionRecommendation(
                type=InterventionType.MUSIC_PLAYLIST,
                title="Iso-Principle Mood Playlist",
                description="A sequence starting at your current mood and stepping toward calm balance.",
                payload={"tracks": [t.__dict__ for t in playlist]},
                timestamp=now
            ))

        return recommendations
