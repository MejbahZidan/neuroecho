"""Interactive terminal runner for testing NeuroEcho interventions."""

import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.intervention_manager import (
    BreathingTechnique,
    InterventionManager,
    InterventionType,
    LocalMusicProvider
)
from core.schemas import AffectLabel, FusedState


def run_terminal_box_breathing(seconds_per_phase: int = 4):
    phases = [
        ("INHALE slowly through your nose", seconds_per_phase, "#"),
        ("HOLD your breath gently", seconds_per_phase, "="),
        ("EXHALE smoothly through your mouth", seconds_per_phase, "~"),
        ("HOLD empty", seconds_per_phase, "."),
    ]

    print("\n========================================================")
    print(f"       GUIDED BOX BREATHING ({seconds_per_phase}s - {seconds_per_phase}s - {seconds_per_phase}s - {seconds_per_phase}s)        ")
    print("========================================================")
    print("Sit comfortably. Press Ctrl+C at any time to finish.\n")

    try:
        for cycle in range(2):
            print(f"\n--- Cycle {cycle + 1}/2 ---")
            for prompt, duration, char in phases:
                for sec in range(1, duration + 1):
                    bar = char * (sec * 6)
                    print(f"\r  {prompt:<36} [{sec}/{duration}s] {bar:<24}", end="", flush=True)
                    time.sleep(1.0)
                print()
        print("\n>> Complete. Deep breath and return to baseline.\n")
    except KeyboardInterrupt:
        print("\n\nBreathing exercise concluded.")


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    manager = InterventionManager(config)

    print("\n--- NeuroEcho Intervention Test Runner ---")
    print("1. Simulate Stressed State -> Trigger Iso-Principle Music")
    print("2. Run Visual Box-Breathing (4-4-4-4)")
    print("3. Simulate Multi-Day Distress -> Trigger Safety Check-in")
    print("------------------------------------------")
    choice = input("Select an option (1, 2, or 3): ").strip()

    if choice == "1":
        fused = FusedState(
            valence=-0.65, arousal=0.75, fatigue=0.4, stress=0.85,
            label=AffectLabel.TENSE_STRESSED, confidence=0.80, contributing_modalities=[], timestamp=time.time()
        )
        recs = manager.evaluate_interventions(fused, posture_score=0.85)
        print(f"\nRecommendations Generated: {len(recs)}")
        for r in recs:
            print(f"\n[{r.title.upper()}] ({r.type.value})")
            print(f"  {r.description}")
            if r.type == InterventionType.MUSIC_PLAYLIST:
                print("  Generated Iso-Principle Track Progression:")
                for i, track in enumerate(r.payload.get("tracks", [])):
                    print(f"    Track {i+1}: '{track['title']}' by {track['artist']} (Valence: {track['valence']:+.2f}, Arousal: {track['arousal']:.2f})")

    elif choice == "2":
        run_terminal_box_breathing(seconds_per_phase=4)

    elif choice == "3":
        manager._distress_days = ["2026-10-30", "2026-10-31", "2026-11-01"]
        fused = FusedState(
            valence=-0.7, arousal=0.8, fatigue=0.8, stress=0.8,
            label=AffectLabel.TENSE_STRESSED, confidence=0.80, contributing_modalities=[], timestamp=time.time()
        )
        recs = manager.evaluate_interventions(fused, posture_score=0.90)
        for r in recs:
            print(f"\n[{r.title.upper()}]")
            print(f"  {r.description}")


if __name__ == "__main__":
    main()
