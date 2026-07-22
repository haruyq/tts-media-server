from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import statistics
import time

from yomogi_onnx import (
    HybridReadingResult,
    YomogiOnnx,
    convert_english_cached,
    convert_hybrid,
    convert_hybrid_async,
)


CASES = {
    "japanese_10": "今日は良い天気です。",
    "english_1": "今日はMinecraftで遊ぶ",
    "english_3": "DiscordでMinecraftとOpenAIの話をする",
    "english_repeat": "MinecraftでMinecraftを遊ぶ",
}


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    case: str
    api: str
    cache: str
    iterations: int
    median_ms: float
    p95_ms: float
    yomogi_median_ms: float
    kanalizer_median_ms: float
    result_total_median_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999))
    return ordered[index]


def _measure_sync(
    reader: YomogiOnnx,
    text: str,
    *,
    iterations: int,
    cold_cache: bool,
) -> BenchmarkRow:
    elapsed: list[float] = []
    results: list[HybridReadingResult] = []
    for _ in range(iterations):
        if cold_cache:
            convert_english_cached.cache_clear()
        started = time.perf_counter()
        result = convert_hybrid(reader, text)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
    return _row("sync", cold_cache, elapsed, results)


async def _measure_async(
    reader: YomogiOnnx,
    text: str,
    *,
    iterations: int,
    cold_cache: bool,
) -> BenchmarkRow:
    elapsed: list[float] = []
    results: list[HybridReadingResult] = []
    for _ in range(iterations):
        if cold_cache:
            convert_english_cached.cache_clear()
        started = time.perf_counter()
        result = await convert_hybrid_async(reader, text)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
    return _row("async", cold_cache, elapsed, results)


def _row(
    api: str,
    cold_cache: bool,
    elapsed: list[float],
    results: list[HybridReadingResult],
) -> BenchmarkRow:
    return BenchmarkRow(
        case="",
        api=api,
        cache="cold" if cold_cache else "warm",
        iterations=len(elapsed),
        median_ms=statistics.median(elapsed),
        p95_ms=_percentile(elapsed, 0.95),
        yomogi_median_ms=statistics.median(
            result.yomogi_elapsed_ms for result in results
        ),
        kanalizer_median_ms=statistics.median(
            result.kanalizer_elapsed_ms for result in results
        ),
        result_total_median_ms=statistics.median(
            result.total_elapsed_ms for result in results
        ),
    )


async def run_benchmark(
    reader: YomogiOnnx,
    *,
    warmup: int,
    iterations: int,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for case, text in CASES.items():
        convert_english_cached.cache_clear()
        for _ in range(warmup):
            convert_hybrid(reader, text)
        for cold_cache in (False, True):
            sync_row = _measure_sync(
                reader,
                text,
                iterations=iterations,
                cold_cache=cold_cache,
            )
            async_row = await _measure_async(
                reader,
                text,
                iterations=iterations,
                cold_cache=cold_cache,
            )
            rows.extend(
                [replace(sync_row, case=case), replace(async_row, case=case)]
            )
    return rows


def _markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        "# Yomogi + Kanalizer hybrid benchmark",
        "",
        "| case | API | cache | median ms | p95 ms | Yomogi ms | Kanalizer ms | total ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.case} | {row.api} | {row.cache} | "
            f"{row.median_ms:.3f} | {row.p95_ms:.3f} | "
            f"{row.yomogi_median_ms:.3f} | {row.kanalizer_median_ms:.3f} | "
            f"{row.result_total_median_ms:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    reader = YomogiOnnx(
        str(args.model_dir),
        intra_op_threads=1,
        inter_op_threads=1,
    )
    rows = asyncio.run(
        run_benchmark(reader, warmup=args.warmup, iterations=args.iterations)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hybrid_benchmark.json"
    markdown_path = args.output_dir / "hybrid_benchmark.md"
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(rows), encoding="utf-8")
    print(_markdown(rows), end="")


if __name__ == "__main__":
    main()
