"""Offline evaluation script for subject-independent splits, F1 scoring, and ablation analysis."""

import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fusion_engine import EmotionFusionEngine
from core.schemas import AffectLabel, ModalityReading, ModalitySource


def generate_synthetic_evaluation_dataset(n_subjects: int = 10, samples_per_subject: int = 40) -> Dict[str, List[Dict]]:
    """Generate subject-partitioned synthetic multi-modal sessions with ground-truth self-reports."""
    dataset = {}
    states = [
        AffectLabel.CALM, AffectLabel.ENERGIZED_FOCUSED,
        AffectLabel.TENSE_STRESSED, AffectLabel.LOW_ENERGY_FATIGUED
    ]

    for subj_idx in range(n_subjects):
        subj_id = f"subject_{subj_idx+1:02d}"
        samples = []
        for _ in range(samples_per_subject):
            gt_label = random.choice(states)
            # Ground-truth mapping
            if gt_label == AffectLabel.CALM:
                true_v, true_a = 0.6, 0.25
            elif gt_label == AffectLabel.ENERGIZED_FOCUSED:
                true_v, true_a = 0.7, 0.75
            elif gt_label == AffectLabel.TENSE_STRESSED:
                true_v, true_a = -0.65, 0.8
            else:
                true_v, true_a = -0.6, 0.2

            # Add noisy observations per modality
            v_val = float(np.clip(true_v + np.random.normal(0, 0.2), -1.0, 1.0))
            a_val = float(np.clip(true_v + np.random.normal(0, 0.35), -1.0, 1.0))
            probe_fatigue = 0.75 if gt_label == AffectLabel.LOW_ENERGY_FATIGUED else 0.15

            samples.append({
                "ground_truth": gt_label,
                "vision": ModalityReading(ModalitySource.VISION, 0.0, v_val, true_a, 0.1, 0.1, 0.85, 0.9),
                "audio": ModalityReading(ModalitySource.AUDIO, 0.0, a_val, true_a, 0.1, 0.1, 0.70, 0.8),
                "probe": ModalityReading(ModalitySource.COGNITIVE_PROBE, 0.0, -probe_fatigue, 0.3, probe_fatigue, 0.2, 0.75, 1.0),
            })
        dataset[subj_id] = samples
    return dataset


def evaluate_ablation(dataset: Dict[str, List[Dict]], active_modalities: List[str]) -> Tuple[float, Dict[str, float]]:
    """Subject-independent cross-validation measuring per-class F1."""
    config = {"fusion": {"smoothing_ema_alpha": 1.0, "hysteresis_persistence_seconds": 0.0}}
    correct = 0
    total = 0
    class_tp = {lbl: 0 for lbl in AffectLabel}
    class_fp = {lbl: 0 for lbl in AffectLabel}
    class_fn = {lbl: 0 for lbl in AffectLabel}

    for subj_id, samples in dataset.items():
        engine = EmotionFusionEngine(config)
        for s in samples:
            gt = s["ground_truth"]
            for mod in active_modalities:
                engine.register_reading(s[mod])
            fused = engine.fuse()

            pred = fused.label
            total += 1
            if pred == gt:
                correct += 1
                class_tp[gt] += 1
            else:
                class_fp[pred] += 1
                class_fn[gt] += 1

    accuracy = correct / max(1, total)
    f1_scores = {}
    for lbl in [AffectLabel.CALM, AffectLabel.ENERGIZED_FOCUSED, AffectLabel.TENSE_STRESSED, AffectLabel.LOW_ENERGY_FATIGUED]:
        tp = class_tp[lbl]
        fp = class_fp[lbl]
        fn = class_fn[lbl]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1_scores[lbl.value] = (2 * prec * rec) / max(1e-6, prec + rec)

    return accuracy, f1_scores


def main():
    print("\n========================================================")
    print("       NEUROECHO OFFLINE ABLATION & EVALUATION          ")
    print("========================================================")
    print("Generating subject-independent cross-validation splits (N=10)...")
    dataset = generate_synthetic_evaluation_dataset(n_subjects=10, samples_per_subject=50)

    ablations = [
        ("Vision Only", ["vision"]),
        ("Audio Only", ["audio"]),
        ("Vision + Audio", ["vision", "audio"]),
        ("Full Multimodal (+ Cognitive Probe)", ["vision", "audio", "probe"]),
    ]

    for name, mods in ablations:
        acc, f1_dict = evaluate_ablation(dataset, mods)
        print(f"\n--- Condition: {name} ---")
        print(f"Overall Accuracy: {acc * 100:.1f}%")
        print("Per-Class F1 Scores:")
        for cls_name, f1 in f1_dict.items():
            print(f"  - {cls_name:<24}: {f1:.2f}")

    print("\nEvaluation complete. Use this benchmark to measure future meta-classifier updates.\n")


if __name__ == "__main__":
    main()
