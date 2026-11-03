"""High-tech, modern PySide6 dashboard with radar visualization, ambient glow, and interactive cards."""

import math
import time
from typing import List

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap, QRadialGradient
)
from PySide6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSlider, QVBoxLayout, QWidget
)

from core.intervention_manager import InterventionRecommendation, InterventionType
from core.schemas import AffectLabel, FusedState, ModalityReading
from ui.widgets.breathing_widget import BreathingDialog
from ui.widgets.probe_dialog import CognitiveProbeDialog


class RadarCircumplexWidget(QWidget):
    """High-tech Radar / Circumplex visualizer with concentric rings and glowing coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 260)
        self.valence = 0.0
        self.arousal = 0.5
        self.glow_pulse = 0.0
        self.glow_dir = 1

        # Animation timer for the glowing coordinate pulse
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_pulse)
        self.timer.start(40)

    def _animate_pulse(self):
        self.glow_pulse += 0.04 * self.glow_dir
        if self.glow_pulse >= 1.0:
            self.glow_dir = -1
        elif self.glow_pulse <= 0.0:
            self.glow_dir = 1
        self.update()

    def set_coords(self, valence: float, arousal: float):
        self.valence = max(-1.0, min(1.0, valence))
        self.arousal = max(0.0, min(1.0, arousal))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        radius = min(cx, cy) - 15.0

        # Background disc
        bg_grad = QRadialGradient(cx, cy, radius)
        bg_grad.setColorAt(0.0, QColor(24, 24, 27))
        bg_grad.setColorAt(1.0, QColor(9, 9, 11))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(39, 39, 42), 1.5))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Concentric Radar Rings
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(63, 63, 70, 70), 1, Qt.DashLine))
        painter.drawEllipse(QPointF(cx, cy), radius * 0.33, radius * 0.33)
        painter.drawEllipse(QPointF(cx, cy), radius * 0.66, radius * 0.66)

        # Crosshairs
        painter.setPen(QPen(QColor(63, 63, 70, 120), 1))
        painter.drawLine(cx - radius, cy, cx + radius, cy)
        painter.drawLine(cx, cy - radius, cx, cy + radius)

        # Quadrant Watermarks
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(239, 68, 68, 120))
        painter.drawText(cx - radius + 10, cy - radius + 22, "TENSE")

        painter.setPen(QColor(6, 182, 212, 120))
        painter.drawText(cx + radius - 64, cy - radius + 22, "FOCUSED")

        painter.setPen(QColor(249, 115, 22, 120))
        painter.drawText(cx - radius + 10, cy + radius - 12, "DEPLETED")

        painter.setPen(QColor(34, 197, 94, 120))
        painter.drawText(cx + radius - 46, cy + radius - 12, "CALM")

        # Coordinate Dot Position
        px = cx + self.valence * (radius - 12)
        py = cy - (self.arousal - 0.5) * 2.0 * (radius - 12)

        # Coordinate Aura Glow
        aura_r = 10.0 + (self.glow_pulse * 6.0)
        aura_grad = QRadialGradient(px, py, aura_r)
        aura_grad.setColorAt(0.0, QColor(59, 130, 246, 200))
        aura_grad.setColorAt(0.7, QColor(59, 130, 246, 50))
        aura_grad.setColorAt(1.0, QColor(59, 130, 246, 0))
        painter.setBrush(QBrush(aura_grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(px, py), aura_r, aura_r)

        # Center Solid Dot
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(QColor(96, 165, 250)))
        painter.drawEllipse(QPointF(px, py), 5.5, 5.5)


class MainWindow(QMainWindow):
    trigger_probe_signal = Signal(str)
    self_report_signal = Signal(int)
    privacy_pause_toggled = Signal(bool)
    delete_data_signal = Signal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.setWindowTitle("NeuroEcho — Affective Computing & Mental Telemetry")
        self.resize(1180, 780)
        self.setStyleSheet("""
            QMainWindow { background-color: #09090b; }
            QLabel { color: #f4f4f5; font-family: 'Segoe UI', system-ui; }
            QFrame.card {
                background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
            QPushButton.action-btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                color: #ffffff;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton.action-btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #2563eb);
            }
            QPushButton.secondary-btn {
                background-color: #1f1f23;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                padding: 8px 14px;
                color: #f4f4f5;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton.secondary-btn:hover {
                background-color: #27272a;
                border-color: #52525b;
            }
            QProgressBar {
                background-color: #1f1f23;
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: right;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #10b981);
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #27272a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: 2px solid #ffffff;
                width: 18px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 9px;
            }
        """)

        self.is_paused = False
        self._build_ui()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        # --- Top Header Bar ---
        header = QHBoxLayout()

        logo_box = QVBoxLayout()
        title = QLabel("NEURO ECHO", self)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("letter-spacing: 2px; color: #ffffff;")
        subtitle = QLabel("AFFECTIVE COMPUTING & REAL-TIME BIO-TELEMETRY", self)
        subtitle.setStyleSheet("color: #71717a; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
        logo_box.addWidget(title)
        logo_box.addWidget(subtitle)
        header.addLayout(logo_box)

        header.addStretch()

        self.badge_active = QLabel("●  SENSORS LIVE", self)
        self.badge_active.setStyleSheet("""
            background-color: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 14px;
            padding: 6px 14px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        header.addWidget(self.badge_active)

        self.btn_pause = QPushButton("Pause Stream", self)
        self.btn_pause.setProperty("class", "secondary-btn")
        self.btn_pause.clicked.connect(self._toggle_pause)
        header.addWidget(self.btn_pause)

        self.btn_delete = QPushButton("Purge Data", self)
        self.btn_delete.setStyleSheet("""
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 8px; padding: 8px 12px; color: #f87171; font-weight: 600; font-size: 11px;
        """)
        self.btn_delete.clicked.connect(self._confirm_delete)
        header.addWidget(self.btn_delete)

        root.addLayout(header)

        # --- Main Body (3 Column Dashboard) ---
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body)

        # ==============================================================
        # COLUMN 1: Visual Sensor Stream & Ergonomics
        # ==============================================================
        col_left = QVBoxLayout()
        body.addLayout(col_left, stretch=4)

        card_cam = QFrame(self)
        card_cam.setProperty("class", "card")
        v_cam = QVBoxLayout(card_cam)
        v_cam.setContentsMargins(14, 14, 14, 14)
        v_cam.setSpacing(10)

        cam_title_box = QHBoxLayout()
        cam_title = QLabel("CAMERA SENSOR STREAM", self)
        cam_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        cam_title.setStyleSheet("color: #a1a1aa; letter-spacing: 0.5px;")
        cam_title_box.addWidget(cam_title)
        cam_title_box.addStretch()

        self.fps_badge = QLabel("30 FPS", self)
        self.fps_badge.setStyleSheet("color: #71717a; font-size: 10px; font-weight: bold;")
        cam_title_box.addWidget(self.fps_badge)
        v_cam.addLayout(cam_title_box)

        self.video_display = QLabel(self)
        self.video_display.setFixedSize(380, 240)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: #09090b; border: 1px solid #27272a; border-radius: 8px;")
        v_cam.addWidget(self.video_display)

        # Telemetry Gauges
        v_cam.addSpacing(4)
        posture_header = QHBoxLayout()
        self.lbl_posture = QLabel("Ergonomic Alignment", self)
        self.lbl_posture.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        self.lbl_posture_val = QLabel("100%", self)
        self.lbl_posture_val.setStyleSheet("color: #10b981; font-weight: bold;")
        posture_header.addWidget(self.lbl_posture)
        posture_header.addStretch()
        posture_header.addWidget(self.lbl_posture_val)
        v_cam.addLayout(posture_header)

        self.bar_posture = QProgressBar(self)
        self.bar_posture.setValue(100)
        self.bar_posture.setTextVisible(False)
        v_cam.addWidget(self.bar_posture)

        self.lbl_eyes = QLabel("Eye Dynamics: 18 blinks/min  |  PERCLOS: 4.2%", self)
        self.lbl_eyes.setStyleSheet("color: #71717a; font-size: 11px;")
        v_cam.addWidget(self.lbl_eyes)

        col_left.addWidget(card_cam)

        # ==============================================================
        # COLUMN 2: Affect Radar & Neural State
        # ==============================================================
        col_center = QVBoxLayout()
        body.addLayout(col_center, stretch=4)

        card_state = QFrame(self)
        card_state.setProperty("class", "card")
        v_state = QVBoxLayout(card_state)
        v_state.setContentsMargins(14, 14, 14, 14)
        v_state.setSpacing(10)

        state_header = QLabel("MULTIMODAL FUSED AFFECT", self)
        state_header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        state_header.setStyleSheet("color: #a1a1aa; letter-spacing: 0.5px;")
        v_state.addWidget(state_header)

        # Main Label Pill
        self.lbl_state = QLabel("NEUTRAL", self)
        self.lbl_state.setAlignment(Qt.AlignCenter)
        self.lbl_state.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.lbl_state.setStyleSheet("""
            color: #60a5fa;
            background-color: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 12px;
            padding: 8px 12px;
            letter-spacing: 1px;
        """)
        v_state.addWidget(self.lbl_state)

        # Radar Circumplex
        self.circumplex = RadarCircumplexWidget(self)
        v_state.addWidget(self.circumplex, alignment=Qt.AlignCenter)

        # Confidence & Attribution
        self.lbl_confidence = QLabel("Certainty Index: 85%", self)
        self.lbl_confidence.setAlignment(Qt.AlignCenter)
        self.lbl_confidence.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        v_state.addWidget(self.lbl_confidence)

        self.lbl_attribution = QLabel("Signals: Vision (60%), Audio (40%)", self)
        self.lbl_attribution.setAlignment(Qt.AlignCenter)
        self.lbl_attribution.setStyleSheet("color: #71717a; font-size: 11px;")
        v_state.addWidget(self.lbl_attribution)

        col_center.addWidget(card_state)

        # ==============================================================
        # COLUMN 3: Smart Interventions & Self-Report
        # ==============================================================
        col_right = QVBoxLayout()
        body.addLayout(col_right, stretch=4)

        card_act = QFrame(self)
        card_act.setProperty("class", "card")
        v_act = QVBoxLayout(card_act)
        v_act.setContentsMargins(14, 14, 14, 14)
        v_act.setSpacing(10)

        act_title = QLabel("RECOVERY & NUDGES", self)
        act_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        act_title.setStyleSheet("color: #a1a1aa; letter-spacing: 0.5px;")
        v_act.addWidget(act_title)

        btn_box = QHBoxLayout()
        self.btn_breathe = QPushButton("🫁  Box Breathe", self)
        self.btn_breathe.setProperty("class", "action-btn")
        self.btn_breathe.clicked.connect(self._launch_breathing)
        btn_box.addWidget(self.btn_breathe)

        self.btn_probe = QPushButton("🎯  Focus Test", self)
        self.btn_probe.setProperty("class", "secondary-btn")
        self.btn_probe.clicked.connect(lambda: self.trigger_probe_signal.emit("stroop"))
        btn_box.addWidget(self.btn_probe)
        v_act.addLayout(btn_box)

        # Recommendations list drawer
        self.lbl_interventions = QLabel("✨ All vitals stable. No active alerts.", self)
        self.lbl_interventions.setWordWrap(True)
        self.lbl_interventions.setStyleSheet("""
            background-color: #18181b;
            border: 1px dashed #27272a;
            border-radius: 8px;
            padding: 12px;
            color: #d4d4d8;
            font-size: 12px;
            line-height: 1.4;
        """)
        v_act.addWidget(self.lbl_interventions)

        col_right.addWidget(card_act)

        # Ground Truth Self-Report Slider Card
        card_self = QFrame(self)
        card_self.setProperty("class", "card")
        v_self = QVBoxLayout(card_self)
        v_self.setContentsMargins(14, 14, 14, 14)
        v_self.setSpacing(8)

        self_title = QLabel("CALIBRATION CHECK-IN", self)
        self_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self_title.setStyleSheet("color: #a1a1aa; letter-spacing: 0.5px;")
        v_self.addWidget(self_title)

        self.lbl_slider_icon = QLabel("😐  3 — Balanced Baseline", self)
        self.lbl_slider_icon.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.lbl_slider_icon.setStyleSheet("color: #60a5fa;")
        v_self.addWidget(self.lbl_slider_icon)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(1, 5)
        self.slider.setValue(3)
        self.slider.valueChanged.connect(self._on_slider_moved)
        v_self.addWidget(self.slider)

        btn_log = QPushButton("Log Ground Truth", self)
        btn_log.setProperty("class", "secondary-btn")
        btn_log.clicked.connect(lambda: self.self_report_signal.emit(self.slider.value()))
        v_self.addWidget(btn_log)

        col_right.addWidget(card_self)

    def _on_slider_moved(self, val: int):
        moods = {
            1: ("😴  1 — Deeply Depleted / Fatigued", "#f87171"),
            2: ("🥱  2 — Low Energy / Strained", "#fb923c"),
            3: ("😐  3 — Balanced Baseline", "#60a5fa"),
            4: ("😊  4 — Calm & Restored", "#4ade80"),
            5: ("⚡  5 — Peak Alertness & Focus", "#22d3ee"),
        }
        text, color = moods.get(val, ("Baseline", "#60a5fa"))
        self.lbl_slider_icon.setText(text)
        self.lbl_slider_icon.setStyleSheet(f"color: {color};")

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.setText("Resume Stream")
            self.badge_active.setText("⏸  STANDBY")
            self.badge_active.setStyleSheet("""
                background-color: rgba(234, 179, 8, 0.12);
                color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3);
                border-radius: 14px; padding: 6px 14px; font-size: 11px; font-weight: 700;
            """)
        else:
            self.btn_pause.setText("Pause Stream")
            self.badge_active.setText("●  SENSORS LIVE")
            self.badge_active.setStyleSheet("""
                background-color: rgba(34, 197, 94, 0.12);
                color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3);
                border-radius: 14px; padding: 6px 14px; font-size: 11px; font-weight: 700;
            """)
        self.privacy_pause_toggled.emit(self.is_paused)

    def _confirm_delete(self):
        reply = QMessageBox.question(
            self, "Purge Local Telemetry",
            "This will permanently erase all local baseline calibrations and distress logs from your drive. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.delete_data_signal.emit()

    def _launch_breathing(self):
        dlg = BreathingDialog(technique=self.config.get("interventions", {}).get("breathing_default", "4-4-4-4"), parent=self)
        dlg.exec()

    @Slot(QImage)
    def update_camera_frame(self, q_img: QImage):
        if not self.is_paused:
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.video_display.width(), self.video_display.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_display.setPixmap(pixmap)

    @Slot(FusedState)
    def update_fused_state(self, state: FusedState):
        self.lbl_state.setText(state.label.value.upper())
        self.circumplex.set_coords(state.valence, state.arousal)
        self.lbl_confidence.setText(f"Certainty Index: {state.confidence * 100:.0f}%")

        theme = {
            AffectLabel.CALM: ("#4ade80", "rgba(34, 197, 94, 0.10)", "rgba(34, 197, 94, 0.3)"),
            AffectLabel.ENERGIZED_FOCUSED: ("#22d3ee", "rgba(6, 182, 212, 0.10)", "rgba(6, 182, 212, 0.3)"),
            AffectLabel.TENSE_STRESSED: ("#f87171", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.3)"),
            AffectLabel.LOW_ENERGY_FATIGUED: ("#fb923c", "rgba(249, 115, 22, 0.10)", "rgba(249, 115, 22, 0.3)"),
            AffectLabel.NEUTRAL: ("#60a5fa", "rgba(59, 130, 246, 0.10)", "rgba(59, 130, 246, 0.3)"),
        }
        color, bg, border = theme.get(state.label, ("#60a5fa", "rgba(59, 130, 246, 0.10)", "rgba(59, 130, 246, 0.3)"))
        self.lbl_state.setStyleSheet(f"""
            color: {color};
            background-color: {bg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 8px 12px;
            letter-spacing: 1px;
        """)

        # Attribution
        expl = [f"{m.title()} ({pct*100:.0f}%)" for m, pct in state.explanation.items() if m != "none"]
        self.lbl_attribution.setText("Signal Share: " + (", ".join(expl) if expl else "Standby"))

    @Slot(ModalityReading)
    def update_vision_telemetry(self, reading: ModalityReading):
        feats = reading.features
        score = feats.get("posture_ergonomic_score", 1.0)
        pct = int(score * 100)
        self.bar_posture.setValue(pct)
        self.lbl_posture_val.setText(f"{pct}%")

        if pct >= 80:
            self.lbl_posture_val.setStyleSheet("color: #4ade80; font-weight: bold;")
        elif pct >= 60:
            self.lbl_posture_val.setStyleSheet("color: #facc15; font-weight: bold;")
        else:
            self.lbl_posture_val.setStyleSheet("color: #f87171; font-weight: bold;")

        bpm = feats.get("blinks_per_min", 0.0)
        perclos = feats.get("perclos_30s", 0.0) * 100
        self.lbl_eyes.setText(f"Eye Dynamics: {bpm:.0f} blinks/min  |  PERCLOS: {perclos:.1f}%")

    @Slot(list)
    def update_recommendations(self, recs: list):
        if not recs:
            self.lbl_interventions.setText("✨ All vitals stable. No active alerts.")
            return

        blocks = []
        for r in recs:
            if r.type == InterventionType.MUSIC_PLAYLIST:
                tracks = r.payload.get("tracks", [])
                t_lines = "".join([f"<li style='margin-bottom: 2px;'><b>{t['title']}</b> — {t['artist']}</li>" for t in tracks[:3]])
                blocks.append(f"<b>🎵 {r.title}</b><ul style='margin: 4px 0 0 16px; padding: 0;'>{t_lines}</ul>")
            elif r.type == InterventionType.POSTURE_RESET:
                blocks.append(f"<b style='color: #fb923c;'>⚠️ {r.title}</b><br>{r.description}")
            elif r.type == InterventionType.SAFETY_CHECKIN:
                blocks.append(f"<b style='color: #f87171;'>🤍 {r.title}</b><br>{r.description}")
            else:
                blocks.append(f"<b>{r.title}</b><br>{r.description}")

        self.lbl_interventions.setText("<hr style='border: 0; border-top: 1px solid #27272a; margin: 6px 0;'>".join(blocks))
