#!/usr/bin/env python3
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


def build_output_path(audio_path: str, extension: str) -> str:
    base, _ = os.path.splitext(audio_path)
    return f"{base}.{extension}"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper.")
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more audio/video files to transcribe.",
    )
    parser.add_argument(
        "--scan-dir",
        default=None,
        help="Scan a directory recursively for common audio/video files.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit path to a model .pt file.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.expanduser("~/.cache/whisper"),
        help="Directory to look for local models (default: ~/.cache/whisper).",
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
    parser.add_argument(
        "--segments-txt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write segments to a .txt file (default: on).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-transcription even if outputs already exist.",
    )
    return parser.parse_args()


def write_segments_txt(result: dict, output_path: str) -> None:
    segments = result.get("segments", [])
    language = result.get("language", "unknown")
    lang_prob = result.get("language_probability", 0.0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(segments)} segments detected\n")
        f.write(f"Language: {language} (prob: {lang_prob:.2f})\n\n")

        for i, seg in enumerate(segments, 1):
            start = f"{seg['start']:.1f}s"
            end = f"{seg['end']:.1f}s"
            text = seg.get("text", "").strip()
            conf = seg.get("avg_logprob", 0)
            f.write(f"[{i:3d}] {start:>6} - {end:<6} | {conf:6.2f} | {text}\n")


def find_media_files(root_dir: str) -> list[str]:
    extensions = {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".flac",
        ".ogg",
        ".webm",
        ".aac",
        ".wma",
        ".mkv",
        ".avi",
        ".mov",
    }
    matches: list[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                matches.append(os.path.join(base, filename))
    return matches


def main() -> int:
    args = parse_args()
    print(get_gpu_info())

    if not args.files and not args.scan_dir:
        print("Error: provide files and/or --scan-dir")
        return 2

    files: list[str] = []
    if args.scan_dir:
        if not os.path.isdir(args.scan_dir):
            print(f"Error: scan dir not found: {args.scan_dir}")
            return 2
        files.extend(find_media_files(args.scan_dir))
    files.extend(args.files)

    seen: set[str] = set()
    unique_files: list[str] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    model_path = args.model_path
    if model_path is None:
        candidate = os.path.join(args.models_dir, f"{args.model}.pt")
        if os.path.exists(candidate):
            model_path = candidate

    model = whisper.load_model(model_path or args.model, device=args.device)

    for audio_file in unique_files:
        if not os.path.exists(audio_file):
            print(f"Skipping missing file: {audio_file}")
            continue

        output_path = build_output_path(audio_file, "json")
        segments_path = build_output_path(audio_file, "txt")
        outputs_exist = os.path.exists(output_path)
        if args.segments_txt:
            outputs_exist = outputs_exist and os.path.exists(segments_path)
        if outputs_exist and not args.force:
            print(f"Skipping existing transcription: {audio_file}")
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved {output_path}")

        if args.segments_txt:
            write_segments_txt(result, segments_path)
            print(f"Saved {segments_path}")

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
