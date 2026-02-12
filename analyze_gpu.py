import argparse
import json
import os
import sys

import torch
import whisper


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        return "GPU: not available"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"GPU: {name} | VRAM: {vram:.1f}GB"


def build_output_path(audio_path: str) -> str:
    base, _ = os.path.splitext(audio_path)
    return f"{base}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper.")
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more audio/video files to transcribe.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name or path (default: large-v3).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 (only valid on CUDA).",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include per-word timestamps (default: off).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(get_gpu_info())

    model = whisper.load_model(args.model, device=args.device)

    for audio_file in args.files:
        if not os.path.exists(audio_file):
            print(f"Skipping missing file: {audio_file}")
            continue

        result = model.transcribe(
            audio_file,
            word_timestamps=args.word_timestamps,
            fp16=args.fp16,
        )

        lang = result.get("language", "unknown")
        lang_prob = result.get("language_probability", 0.0)
        print(f"\nFile: {audio_file}")
        print(f"Language: {lang} (prob: {lang_prob:.2f})")
        print("Full transcript length:", len(result.get("text", "")))

        output_path = build_output_path(audio_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved {output_path}")

        segments = result.get("segments", [])
        if segments:
            avg_duration = sum(s["end"] - s["start"] for s in segments) / len(segments)
            print(f"Segments: {len(segments)} | Avg duration: {avg_duration:.0f}s")
        else:
            print("Segments: 0")

        if args.word_timestamps and segments and "words" in segments[0]:
            print("First segment words:")
            for word in segments[0]["words"][:10]:
                print(
                    f"  '{word['word']}' "
                    f"[{word['start']:.1f}-{word['end']:.1f}s] "
                    f"conf: {word['probability']:.2f}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
