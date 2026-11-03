"""Diagnostic script for NeuroEcho audio processing (Microphone or Audio File)."""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.audio_processor import AudioProcessor


def process_audio_stream(processor: AudioProcessor, source_type: str, file_path: str = None):
    sr = processor.sample_rate
    chunk_len = int(processor.chunk_duration * sr)

    print("\n--- NeuroEcho Audio Diagnostic Mode ---")
    print(f"Sampling Rate: {sr} Hz | Window: {processor.chunk_duration}s")
    print("Commands:")
    print("  [Ctrl+C] - Exit")
    print("---------------------------------------\n")

    if source_type == "file":
        import librosa
        print(f">> Loading audio file: {file_path}")
        audio_data, _ = librosa.load(file_path, sr=sr, mono=True)
        total_chunks = len(audio_data) // chunk_len
        
        if total_chunks == 0:
            print("File is shorter than 5 seconds.")
            return

        for i in range(total_chunks):
            chunk = audio_data[i * chunk_len : (i + 1) * chunk_len]
            print(f"\n--- Analyzing Chunk {i+1}/{total_chunks} ---")
            reading = processor.process_buffer(chunk)
            print_reading(reading)
            time.sleep(1.0)
    else:
        import sounddevice as sd
        print(">> Listening on default microphone... Speak continuously for 5 seconds.")
        try:
            while True:
                # Record 5s in RAM
                audio_buffer = sd.rec(chunk_len, samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                chunk = audio_buffer.flatten()

                reading = processor.process_buffer(chunk)
                print_reading(reading)
        except KeyboardInterrupt:
            print("\nStopped.")


def print_reading(reading):
    f = reading.features
    print(f"Speech Detected: {f.get('speech_detected', False)} (Ratio: {f.get('speech_ratio', 0.0)*100:.1f}%)")
    print(f"Signal Quality : {reading.quality*100:.1f}% (SNR: {f.get('snr_db', 0.0):.1f} dB)")
    if f.get("speech_detected", False):
        print(f"Top SER Emotion: {f.get('top_emotion', 'N/A')}")
        print(f"F0 Mean        : {f.get('f0_mean', 0.0):.1f} Hz (Z-Score: {f.get('f0_z', 0.0):+.2f})")
        print(f"Pause Ratio    : {f.get('pause_ratio', 0.0)*100:.1f}%")
        print(f"Absolutist Flag: {f.get('hopeless_flag', False)} (Count: {f.get('absolutist_count', 0)})")
        print(f"AFFECT VALUES  -> Valence: {reading.valence:+.2f} | Arousal: {reading.arousal:.2f} | Fatigue: {reading.fatigue:.2f} | Stress: {reading.stress:.2f}")
        print(f"Confidence     : {reading.confidence*100:.1f}%")
    else:
        print(">> (Silence / Background noise; no speech gated)")


def main():
    parser = argparse.ArgumentParser(description="NeuroEcho Audio Diagnostic")
    parser.add_argument("--file", type=str, default=None, help="Path to .wav or .mp3 file")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing AudioProcessor (checking dependencies)...")
    processor = AudioProcessor(config)

    if args.file:
        process_audio_stream(processor, source_type="file", file_path=args.file)
    else:
        process_audio_stream(processor, source_type="mic")


if __name__ == "__main__":
    main()
