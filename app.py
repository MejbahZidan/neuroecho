"""Application entrypoint coordinating capture threads, bounded queues, and the PySide6 UI."""

import os
import shutil
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMessageBox
import yaml

from core.audio_processor import AudioProcessor
from core.cognitive_probes import CognitiveProbeEngine, ProbeType
from core.fusion_engine import EmotionFusionEngine
from core.intervention_manager import InterventionManager
from core.schemas import FusedState, ModalityReading
from core.vision_processor import VisionProcessor
from ui.main_window import MainWindow
from ui.widgets.probe_dialog import CognitiveProbeDialog


class VisionWorker(QThread):
    frame_processed = Signal(QImage)
    reading_ready = Signal(ModalityReading)

    def __init__(self, processor: VisionProcessor, video_source=0):
        super().__init__()
        self.processor = processor
        self.video_source = video_source
        self.running = True
        self.paused = False

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened() and isinstance(self.video_source, int):
            # Fallback to local sample video if no hardware webcam exists
            fallback = "sample.mp4"
            if os.path.exists(fallback):
                cap = cv2.VideoCapture(fallback)

        while self.running:
            if self.paused or not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                if isinstance(self.video_source, str) or os.path.exists("sample.mp4"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            # Process through core VisionProcessor
            reading = self.processor.process_frame(frame)
            self.reading_ready.emit(reading)

            # Convert BGR frame to QImage for non-blocking UI rendering
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_processed.emit(q_img)

            # Cap vision processing loop at ~30 FPS
            time.sleep(0.033)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class AudioWorker(QThread):
    reading_ready = Signal(ModalityReading)

    def __init__(self, processor: AudioProcessor):
        super().__init__()
        self.processor = processor
        self.running = True
        self.paused = False

    def run(self):
        try:
            import sounddevice as sd
            sr = self.processor.sample_rate
            chunk_len = int(self.processor.chunk_duration * sr)
        except Exception:
            return  # Degrade gracefully if sounddevice is unavailable

        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            try:
                # Record 5s buffer in RAM
                buf = sd.rec(chunk_len, samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                if not self.running:
                    break
                chunk = buf.flatten()
                reading = self.processor.process_buffer(chunk)
                if reading.confidence > 0.0:
                    self.reading_ready.emit(reading)
            except Exception:
                time.sleep(1.0)

    def stop(self):
        self.running = False
        self.wait()


def show_consent_dialog(parent=None) -> bool:
    consent_file = os.path.join("data", ".consent_granted")
    if os.path.exists(consent_file):
        return True

    msg = QMessageBox(parent)
    msg.setWindowTitle("NeuroEcho - Privacy & Wellbeing Consent")
    msg.setText(
        "<h3>Welcome to NeuroEcho</h3>"
        "<p>NeuroEcho is a <b>personal wellbeing tool</b>, NOT a medical or diagnostic device. "
        "It estimates your stress, fatigue, and posture locally on your computer.</p>"
        "<p><b>Privacy Commitments:</b></p>"
        "<ul>"
        "<li>All camera and microphone analysis runs entirely in local device RAM.</li>"
        "<li>No raw video or audio is ever written to disk or sent across the internet.</li>"
        "<li>You can pause monitoring or delete stored baselines at any time.</li>"
        "</ul>"
        "<p>Do you agree to continue?</p>"
    )
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)
    res = msg.exec()

    if res == QMessageBox.Yes:
        os.makedirs("data", exist_ok=True)
        with open(consent_file, "w") as f:
            f.write("consented")
        return True
    return False


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    app = QApplication(sys.argv)

    if not show_consent_dialog():
        sys.exit(0)

    # Core logic instantiation (UI-agnostic)
    vision_proc = VisionProcessor(config)
    audio_proc = AudioProcessor(config, lazy_load_models=False)
    probe_engine = CognitiveProbeEngine(config)
    fusion_engine = EmotionFusionEngine(config)
    intervention_mgr = InterventionManager(config)

    # UI instantiation
    window = MainWindow(config)

    # Capture Workers
    v_source = config.get("vision", {}).get("camera_index", 0)
    vision_worker = VisionWorker(vision_proc, video_source=v_source)
    audio_worker = AudioWorker(audio_proc)

    # Signal Wiring
    vision_worker.frame_processed.connect(window.update_camera_frame)

    def on_vision_reading(reading: ModalityReading):
        window.update_vision_telemetry(reading)
        fusion_engine.register_reading(reading)
        fused = fusion_engine.fuse()
        window.update_fused_state(fused)

        posture_score = reading.features.get("posture_ergonomic_score", 1.0)
        recs = intervention_mgr.evaluate_interventions(fused, posture_score=posture_score)
        window.update_recommendations(recs)

        # Evaluate probe trigger rule
        if probe_engine.trigger_manager.record_reading(fused.fatigue, fused.stress, fused.confidence):
            dlg = CognitiveProbeDialog(probe_engine, task=ProbeType.STROOP, parent=window)
            if dlg.exec() == CognitiveProbeDialog.Accepted and dlg.modality_reading:
                fusion_engine.register_reading(dlg.modality_reading)
                window.update_fused_state(fusion_engine.fuse())

    def on_audio_reading(reading: ModalityReading):
        fusion_engine.register_reading(reading)
        fused = fusion_engine.fuse()
        window.update_fused_state(fused)

    vision_worker.reading_ready.connect(on_vision_reading)
    audio_worker.reading_ready.connect(on_audio_reading)

    def on_self_report(score: int):
        fused = fusion_engine.record_self_report(score)
        window.update_fused_state(fused)

    def on_pause_toggled(is_paused: bool):
        vision_worker.paused = is_paused
        audio_worker.paused = is_paused

    def on_delete_data():
        if os.path.exists("data"):
            for fname in os.listdir("data"):
                if fname != ".consent_granted":
                    fpath = os.path.join("data", fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)

    def on_trigger_probe_request(task_str: str):
        task = ProbeType.STROOP if task_str == "stroop" else ProbeType.PVT
        dlg = CognitiveProbeDialog(probe_engine, task=task, parent=window)
        if dlg.exec() == CognitiveProbeDialog.Accepted and dlg.modality_reading:
            fusion_engine.register_reading(dlg.modality_reading)
            window.update_fused_state(fusion_engine.fuse())

    window.self_report_signal.connect(on_self_report)
    window.privacy_pause_toggled.connect(on_pause_toggled)
    window.delete_data_signal.connect(on_delete_data)
    window.trigger_probe_signal.connect(on_trigger_probe_request)

    # Start Workers
    vision_worker.start()
    audio_worker.start()

    window.show()
    exit_code = app.exec()

    # Clean shutdown
    vision_worker.stop()
    audio_worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
