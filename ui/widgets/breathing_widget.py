"""Animated visual widget for Box Breathing (4-4-4-4) and Relaxing Breathing (4-7-8)."""

import math
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class BreathingDialog(QDialog):
    def __init__(self, technique: str = "4-4-4-4", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guided Breathing - NeuroEcho")
        self.setFixedSize(400, 480)
        self.setStyleSheet("background-color: #18181b; color: #f4f4f5;")

        if technique == "4-7-8":
            self.phases = [("Inhale (Nose)", 4.0), ("Hold", 7.0), ("Exhale (Mouth)", 8.0), ("Rest", 1.0)]
        else:
            self.phases = [("Inhale", 4.0), ("Hold", 4.0), ("Exhale", 4.0), ("Hold", 4.0)]

        self.phase_idx = 0
        self.phase_time_elapsed = 0.0
        self.scale_factor = 0.3

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30 FPS

        layout = QVBoxLayout(self)
        layout.addStretch()

        self.label_prompt = QLabel(self.phases[0][0], self)
        self.label_prompt.setAlignment(Qt.AlignCenter)
        self.label_prompt.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(self.label_prompt)

        self.label_timer = QLabel("4.0s", self)
        self.label_timer.setAlignment(Qt.AlignCenter)
        self.label_timer.setFont(QFont("Segoe UI", 12))
        layout.addWidget(self.label_timer)

        layout.addSpacing(220)

        btn_close = QPushButton("Finish Exercise", self)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #27272a; border: 1px solid #3f3f46;
                border-radius: 6px; padding: 8px; color: #fafafa; font-weight: bold;
            }
            QPushButton:hover { background-color: #3f3f46; }
        """)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _tick(self):
        dt = 0.033
        self.phase_time_elapsed += dt
        curr_phase_name, curr_phase_dur = self.phases[self.phase_idx]

        rem = max(0.0, curr_phase_dur - self.phase_time_elapsed)
        self.label_timer.setText(f"{rem:.1f}s")
        self.label_prompt.setText(curr_phase_name)

        progress = min(1.0, self.phase_time_elapsed / curr_phase_dur)
        if "Inhale" in curr_phase_name:
            self.scale_factor = 0.3 + 0.6 * progress
        elif "Exhale" in curr_phase_name:
            self.scale_factor = 0.9 - 0.6 * progress

        if self.phase_time_elapsed >= curr_phase_dur:
            self.phase_time_elapsed = 0.0
            self.phase_idx = (self.phase_idx + 1) % len(self.phases)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx, cy = self.width() / 2.0, 210.0
        max_r = 90.0
        r = max_r * self.scale_factor

        painter.setPen(QPen(QColor(59, 130, 246, 60), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), max_r, max_r)

        painter.setPen(QPen(QColor(96, 165, 250), 3))
        painter.setBrush(QBrush(QColor(59, 130, 246, 120)))
        painter.drawEllipse(QPointF(cx, cy), r, r)
