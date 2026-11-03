import argparse
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.vision_processor import VisionProcessor


def main():
    parser = argparse.ArgumentParser(description="NeuroEcho Vision Diagnostic Viewer")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of live webcam")
    args = parser.parse_args()

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Initializing VisionProcessor (downloading models if needed)...")
    processor = VisionProcessor(config)

    # Determine input source: video file or webcam
    if args.video:
        print(f"Loading video from: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        cam_idx = config.get("vision", {}).get("camera_index", 0)
        cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        print("Error: No webcam found and no video file specified.")
        print("Tip: You can pass a video file to test: python scripts\\debug_vision.py --video sample.mp4")
        return

    print("\n--- NeuroEcho Vision Diagnostic Mode ---")
    print("Commands:")
    print("  [C] - Start 10-second Calibration")
    print("  [Q] - Quit")
    print("----------------------------------------\n")

    calib_start_time = None
    fps_time = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                # Loop video continuously
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        now = time.time()
        fps_count += 1
        if now - fps_time >= 1.0:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        if processor.calibrating and calib_start_time:
            if now - calib_start_time >= 10.0:
                processor.finish_calibration()
                calib_start_time = None
                print(">> Calibration finished.")

        t0 = time.perf_counter()
        reading = processor.process_frame(frame, timestamp=now)
        proc_latency_ms = (time.perf_counter() - t0) * 1000.0

        hud = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(hud, (0, 0), (w, 85), (20, 20, 20), -1)
        status_text = "CALIBRATING..." if processor.calibrating else ("CALIBRATED" if processor.calibrated else "UNCALIBRATED")
        status_color = (0, 165, 255) if processor.calibrating else ((0, 255, 0) if processor.calibrated else (0, 0, 255))
        cv2.putText(hud, f"Status: {status_text}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(hud, f"FPS: {fps:.1f} | Pipeline Latency: {proc_latency_ms:.1f} ms", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        feats = reading.features
        if feats.get("face_detected", False):
            cv2.putText(hud, f"EAR: {feats['ear']:.3f} (Thresh: {processor.baselines['ear_threshold']:.3f})", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"Blinks/min: {feats['blinks_per_min']:.1f}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"Mean Blink Dur: {feats['mean_blink_duration_ms']:.0f} ms", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"PERCLOS: {feats['perclos_30s'] * 100:.1f}%", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"Posture Score: {feats['posture_ergonomic_score'] * 100:.0f}%", (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.putText(hud, f"Valence: {reading.valence:+.2f}", (w - 220, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"Arousal: {reading.arousal:.2f}", (w - 220, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(hud, f"Fatigue: {reading.fatigue:.2f}", (w - 220, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
            cv2.putText(hud, f"Stress:  {reading.stress:.2f}", (w - 220, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            cv2.putText(hud, f"Confidence: {reading.confidence * 100:.0f}%", (w - 220, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(hud, "NO FACE DETECTED", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("NeuroEcho Vision Diagnostic HUD", hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not processor.calibrating:
            print(">> Starting calibration (10 seconds)...")
            processor.start_calibration()
            calib_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
