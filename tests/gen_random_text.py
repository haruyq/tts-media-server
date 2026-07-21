import argparse
import random
import string

JAPANESE_CHARS = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねの"
    "はひふへほまみむめもやゆよらりるれろわをん"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
    "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "日本語音声合成文章試験確認生成文字情報時間今日明日"
)

PUNCTUATION = "、。！？"

def generate_random_text(length: int, mode: str) -> str:
    if length < 0:
        raise ValueError("文字数は0以上で指定してください。")

    if mode == "ascii":
        chars = string.ascii_letters + string.digits
    elif mode == "japanese":
        chars = JAPANESE_CHARS
    elif mode == "speech":
        chars = JAPANESE_CHARS + PUNCTUATION
    else:
        raise ValueError(f"不明なモードです: {mode}")

    return "".join(random.choices(chars, k=length))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定文字数のランダム文字列を生成します。"
    )
    parser.add_argument(
        "length",
        type=int,
        help="生成する文字数",
    )
    parser.add_argument(
        "--mode",
        choices=["ascii", "japanese", "speech"],
        default="speech",
        help="文字種。既定値: speech",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="乱数シード。同じ値なら同じ文字列を生成します。",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        print(generate_random_text(args.length, args.mode))
    except ValueError as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()