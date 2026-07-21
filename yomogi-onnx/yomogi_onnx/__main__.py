import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .runtime import YomogiOnnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Yomogi v1.4 ONNX inference")
    parser.add_argument("text", help="Japanese text to read")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).parents[1] / "dist"),
        help="Directory containing exported Yomogi artifacts",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--max-length", type=int, default=500)
    args = parser.parse_args()

    reader = YomogiOnnx(args.model_dir, max_length=args.max_length)
    result = reader.infer(args.text)
    if args.json:
        payload = asdict(result)
        payload.update(
            {
                "read_katakana": result.read_katakana,
                "pron_katakana": result.pron_katakana,
                "pron_hiragana": result.pron_hiragana,
                "tts_text": result.tts_text,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(result.tts_text)


if __name__ == "__main__":
    main()
