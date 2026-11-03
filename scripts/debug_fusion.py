"""Interactive terminal simulator for testing EmotionFusionEngine behavior."""

import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fusion_engine import EmotionFusionEngine
from core.schemas import ModalityReading, ModalitySource


def print_dashboard(state):
    os.system("cls" if os.name == "nt" else "clear")
    print("=================================================================")
    print("           NEUROECHO EMOTION FUSION ENGINE - SIMULATOR           ")
    print("=================================================================")
    print(f"CURRENT AFFECT LABEL : >>>  {state.label.value.upper()}  <<<")
    print(f"Overall Confidence   : {state.confidence * 100:.1f}%")
    print("-----------------------------------------------------------------")
    print(f"Smoothed Valence (X) : [{state.valence:+.2f}]  (-1.0 = Distress, +1.0 = Positive)")
    print(f"Smoothed Arousal (Y) : [{state.arousal:.2f}]   (0.0 = Lethargic, 1.0 = High Energy)")
    print(f"Fatigue Index        : [{state.fatigue:.2f}]")
    print(f"Stress Index         : [{state.stress:.2f}]")
    print("-----------------------------------------------------------------")
    print("Contributing Channels & Weights:")
    for mod, w in state.explanation.items():
        bar = "#" * int(w * 20)
        print(f"  - {mod.upper():<16} : {w * 100:5.1f}% | {bar}")
    print("=================================================================")
    print("Scenario in progress (watching EMA smoothing & hysteresis)...")


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    engine = EmotionFusionEngine(config)
    print("Starting simulation timeline...")
    t = 100.0

    scenarios = [
        ("Step 1: Calm working session (Vision only)", 4, {"v_val": 0.5, "v_ar": 0.3, "v_conf": 0.85, "fatigue": 0.1, "stress": 0.1}),
        ("Step 2: Audio noise spike (1 second flick - should NOT switch label)", 2, {"v_val": 0.5, "v_ar": 0.3, "v_conf": 0.85, "a_val": -0.9, "a_ar": 0.9, "a_conf": 0.3}),
        ("Step 3: Sustained afternoon fatigue enters (Eye droop + slow cadence)", 5, {"v_val": -0.4, "v_ar": 0.2, "v_conf": 0.85, "fatigue": 0.75, "stress": 0.2}),
    ]

    for desc, steps, params in scenarios:
        for s in range(steps):
            t += 1.0
            v_val = params.get("v_val", 0.0)
            v_ar = params.get("v_ar", 0.5)
            fatigue = params.get("fatigue", 0.1)
            stress = params.get("stress", 0.1)
            v_conf = params.get("v_conf", 0.8)

            engine.register_reading(ModalityReading(
                source=ModalitySource.VISION, timestamp=t,
                valence=v_val, arousal=v_ar, fatigue=fatigue, stress=stress,
                confidence=v_conf, quality=0.9
            ))

            if "a_val" in params:
                engine.register_reading(ModalityReading(
                    source=ModalitySource.AUDIO, timestamp=t,
                    valence=params["a_val"], arousal=params["a_ar"], fatigue=fatigue, stress=stress,
                    confidence=params["a_conf"], quality=0.8
                ))

            state = engine.fuse(timestamp=t)
            print_dashboard(state)
            print(f">> Scenario: {desc} (Tick {s+1}/{steps})")
            time.sleep(0.6)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
