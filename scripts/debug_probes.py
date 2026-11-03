"""Interactive terminal runner for Stroop and PVT cognitive probes."""

import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cognitive_probes import CognitiveProbeEngine, ProbeType, TrialResult


def run_interactive_pvt(engine: CognitiveProbeEngine, n_trials: int = 5):
    print("\n========================================================")
    print("       PSYCHOMOTOR VIGILANCE TASK (PVT) - BRIEF        ")
    print("========================================================")
    print("Instructions:")
    print("  When you see '>>> PRESS ENTER NOW! <<<', press [ENTER]")
    print("  as fast as you possibly can. Do not guess ahead.")
    print("========================================================\n")
    input("Press [ENTER] when ready to start...")

    delays = engine.generate_pvt_delays(n_trials=n_trials, min_delay_s=1.5, max_delay_s=3.5)
    trials = []

    for i, delay in enumerate(delays):
        print(f"\n[Trial {i+1}/{n_trials}] Focus on the screen...")
        time.sleep(delay)

        print("\n>>> PRESS ENTER NOW! <<<")
        t_start = time.perf_counter()
        input()
        rt_ms = (time.perf_counter() - t_start) * 1000.0

        is_anticipation = rt_ms < 100.0
        is_lapse = rt_ms > 500.0

        if is_anticipation:
            print(f"  [!] False Start / Anticipation ({rt_ms:.0f} ms < 100 ms)")
        elif is_lapse:
            print(f"  [X] Attentional Lapse! Slower than 500 ms ({rt_ms:.0f} ms)")
        else:
            print(f"  [✓] Good reaction: {rt_ms:.0f} ms")

        trials.append(TrialResult(
            stimulus="TARGET", response="ENTER", target="ENTER",
            reaction_time_ms=rt_ms, correct=not is_anticipation,
            is_anticipation=is_anticipation, is_lapse=is_lapse
        ))

    session_res, reading = engine.evaluate_session(ProbeType.PVT, trials)
    print_results(session_res, reading)


def run_interactive_stroop(engine: CognitiveProbeEngine, n_trials: int = 6):
    print("\n========================================================")
    print("                 STROOP TASK - BRIEF                    ")
    print("========================================================")
    print("Instructions:")
    print("  Identify the INK COLOR of the word, NOT what it spells!")
    print("  Keys: [R]=RED  [G]=GREEN  [B]=BLUE  [Y]=YELLOW")
    print("========================================================\n")
    input("Press [ENTER] when ready to start...")

    stimuli = engine.generate_stroop_trials(n_trials=n_trials)
    key_map = {"R": "RED", "G": "GREEN", "B": "BLUE", "Y": "YELLOW"}
    trials = []

    for i, stim in enumerate(stimuli):
        print("\n--------------------------------------------------------")
        print(f"Trial {i+1}/{n_trials}:")
        print(f"   Word text: [{stim['word']}]  <-- INK COLOR: *** {stim['color']} ***")
        print("Type R, G, B, or Y and press [ENTER]:")

        t_start = time.perf_counter()
        user_key = input("Your answer: ").strip().upper()
        rt_ms = (time.perf_counter() - t_start) * 1000.0

        user_choice = key_map.get(user_key, "INVALID")
        correct = (user_choice == stim["target"])
        is_anticipation = rt_ms < 100.0
        is_lapse = rt_ms > 1500.0

        status = "CORRECT" if correct else f"WRONG (was {stim['target']})"
        print(f"  [{status}] in {rt_ms:.0f} ms")

        trials.append(TrialResult(
            stimulus=f"{stim['word']}_{stim['color']}",
            response=user_choice,
            target=stim["target"],
            reaction_time_ms=rt_ms,
            correct=correct,
            is_anticipation=is_anticipation,
            is_lapse=is_lapse
        ))

    session_res, reading = engine.evaluate_session(ProbeType.STROOP, trials)
    print_results(session_res, reading)


def print_results(res, reading):
    print("\n================ SESSION RESULTS ================")
    print(f"Probe Type             : {res.probe_type.value.upper()}")
    print(f"Median Reaction Time   : {res.median_rt_ms:.1f} ms (Mean: {res.mean_rt_ms:.1f} ms)")
    print(f"Accuracy               : {res.accuracy * 100:.1f}%")
    print(f"Lapse Rate             : {res.lapse_rate * 100:.1f}%")
    print(f"Anticipation Rate      : {res.anticipation_rate * 100:.1f}%")
    print(f"Cognitive Fatigue Index: {res.cognitive_fatigue_index:.2f} (0=Alert, 1=Exhausted)")
    print("-------------------------------------------------")
    print(f"PRODUCED ModalityReading:")
    print(f"  Valence : {reading.valence:+.2f}")
    print(f"  Arousal : {reading.arousal:.2f}")
    print(f"  Fatigue : {reading.fatigue:.2f}")
    print(f"  Stress  : {reading.stress:.2f}")
    print(f"  Confidence (Bounded) : {reading.confidence * 100:.1f}%")
    print("=================================================\n")


def main():
    parser = argparse.ArgumentParser(description="NeuroEcho Interactive Cognitive Probes")
    parser.add_argument("--task", choices=["pvt", "stroop"], default="pvt", help="Task type to run")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    engine = CognitiveProbeEngine(config)

    if args.task == "pvt":
        run_interactive_pvt(engine, n_trials=args.trials)
    else:
        run_interactive_stroop(engine, n_trials=args.trials)


if __name__ == "__main__":
    main()
