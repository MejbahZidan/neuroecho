"""Interactive PySide6 dialog for Stroop and PVT cognitive verification probes."""

import random
import time
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from core.cognitive_probes import CognitiveProbeEngine, ProbeType, TrialResult


class CognitiveProbeDialog(QDialog):
    COLOR_MAP = {
        "RED": "#ef4444",
        "GREEN": "#22c55e",
        "BLUE": "#3b82f6",
        "YELLOW": "#eab308"
    }

    def __init__(self, engine: CognitiveProbeEngine, task: ProbeType = ProbeType.STROOP, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.task = task
        self.session_result = None
        self.modality_reading = None
        self.trials: list[TrialResult] = []

        self.setWindowTitle(f"Quick Focus Verification - {self.task.value.upper()}")
        self.setFixedSize(500, 360)
        self.setStyleSheet("background-color: #18181b; color: #f4f4f5;")

        self.current_trial_idx = 0
        self.trial_start_time = 0.0

        layout = QVBoxLayout(self)

        self.label_info = QLabel("Identify the INK COLOR (not the word text)", self)
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self.label_info)

        self.label_display = QLabel("READY", self)
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.label_display.setStyleSheet("margin: 30px;")
        layout.addWidget(self.label_display)

        self.btn_layout = QHBoxLayout()
        layout.addLayout(self.btn_layout)

        if self.task == ProbeType.STROOP:
            self.stimuli = self.engine.generate_stroop_trials(n_trials=8)
            for color_name, hex_code in self.COLOR_MAP.items():
                btn = QPushButton(f"{color_name} ({color_name[0]})", self)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #27272a; border: 2px solid {hex_code};
                        border-radius: 6px; padding: 10px; color: #fafafa; font-weight: bold;
                    }}
                    QPushButton:hover {{ background-color: {hex_code}; color: #000; }}
                """)
                btn.clicked.connect(lambda ch=False, c=color_name: self._handle_stroop_choice(c))
                self.btn_layout.addWidget(btn)
                # Shortcuts: R, G, B, Y
                QShortcut(QKeySequence(color_name[0]), self).activated.connect(lambda c=color_name: self._handle_stroop_choice(c))
        else:
            self.pvt_delays = self.engine.generate_pvt_delays(n_trials=6)
            self.btn_react = QPushButton("PRESS SPACEBAR OR CLICK HERE", self)
            self.btn_react.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6; border-radius: 8px; padding: 14px;
                    font-size: 14px; font-weight: bold; color: white;
                }
                QPushButton:hover { background-color: #2563eb; }
            """)
            self.btn_react.clicked.connect(self._handle_pvt_hit)
            QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self._handle_pvt_hit)
            self.btn_layout.addWidget(self.btn_react)

        QTimer.singleShot(600, self._next_trial)

    def _next_trial(self):
        if self.task == ProbeType.STROOP:
            if self.current_trial_idx >= len(self.stimuli):
                self._finish_session()
                return
            stim = self.stimuli[self.current_trial_idx]
            self.label_display.setText(stim["word"])
            self.label_display.setStyleSheet(f"color: {self.COLOR_MAP[stim['color']]}; margin: 30px;")
            self.trial_start_time = time.perf_counter()
        else:
            if self.current_trial_idx >= len(self.pvt_delays):
                self._finish_session()
                return
            self.label_display.setText("WAIT...")
            self.label_display.setStyleSheet("color: #71717a; margin: 30px;")
            delay_ms = int(self.pvt_delays[self.current_trial_idx] * 1000)
            QTimer.singleShot(delay_ms, self._show_pvt_target)

    def _show_pvt_target(self):
        self.label_display.setText("HIT NOW!")
        self.label_display.setStyleSheet("color: #ef4444; margin: 30px;")
        self.trial_start_time = time.perf_counter()

    def _handle_stroop_choice(self, chosen_color: str):
        if self.trial_start_time == 0:
            return
        rt_ms = (time.perf_counter() - self.trial_start_time) * 1000.0
        stim = self.stimuli[self.current_trial_idx]
        correct = (chosen_color == stim["target"])

        self.trials.append(TrialResult(
            stimulus=f"{stim['word']}_{stim['color']}", response=chosen_color, target=stim["target"],
            reaction_time_ms=rt_ms, correct=correct,
            is_anticipation=rt_ms < 100.0, is_lapse=rt_ms > 1500.0
        ))
        self.current_trial_idx += 1
        self._next_trial()

    def _handle_pvt_hit(self):
        if self.trial_start_time == 0:
            return
        rt_ms = (time.perf_counter() - self.trial_start_time) * 1000.0
        is_anti = rt_ms < 100.0
        self.trials.append(TrialResult(
            stimulus="PVT_TARGET", response="HIT", target="HIT",
            reaction_time_ms=rt_ms, correct=not is_anti,
            is_anticipation=is_anti, is_lapse=rt_ms > 500.0
        ))
        self.trial_start_time = 0
        self.current_trial_idx += 1
        self._next_trial()

    def _finish_session(self):
        self.session_result, self.modality_reading = self.engine.evaluate_session(self.task, self.trials)
        self.accept()
