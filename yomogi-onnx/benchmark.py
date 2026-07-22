from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import psutil


LENGTHS = (10, 30, 100, 300)
BASE_TEXT = "今日は人気の商品を紹介します。日本橋へ向かい、大人しく待ってください。"


def input_for_length(length: int) -> str:
    return (BASE_TEXT * ((length // len(BASE_TEXT)) + 1))[:length]


def percentiles(values: list[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
    }


def artifact_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def build_reader(
    variant: str,
    source_dir: Path,
    model_dir: Path,
) -> tuple[Any, list[Path]]:
    if variant == "pytorch_cpu":
        from reference_model import TorchYomogiReference

        return TorchYomogiReference(source_dir), [source_dir / "model/model.pt"]

    from yomogi_onnx.runtime import YomogiOnnx

    common = [
        model_dir / "dictionary.tsv",
        model_dir / "input_tokens.tsv",
        model_dir / "surface_vocab.tsv",
        model_dir / "model_meta.json",
    ]
    if variant == "encoder_fp32_memory":
        return YomogiOnnx(str(model_dir)), common + [
            model_dir / "yomogi_encoder_fp32.onnx",
            model_dir / "candidate_weight_fp32.npy",
            model_dir / "candidate_bias_fp32.npy",
        ]
    if variant == "encoder_fp32_mmap":
        return YomogiOnnx(str(model_dir), parameter_loading="mmap"), common + [
            model_dir / "yomogi_encoder_fp32.onnx",
            model_dir / "candidate_weight_fp32.npy",
            model_dir / "candidate_bias_fp32.npy",
        ]
    if variant == "full_fp32":
        return YomogiOnnx(
            str(model_dir),
            model_filename="yomogi_full_fp32.onnx",
            full_model=True,
        ), common + [model_dir / "yomogi_full_fp32.onnx"]
    if variant == "encoder_int8_experimental":
        return YomogiOnnx(
            str(model_dir),
            model_filename="experimental/yomogi_encoder_int8.onnx",
        ), common + [
            model_dir / "experimental/yomogi_encoder_int8.onnx",
            model_dir / "candidate_weight_fp32.npy",
            model_dir / "candidate_bias_fp32.npy",
        ]
    raise ValueError(f"Unknown benchmark variant: {variant}")


def run_worker(
    variant: str,
    source_dir: Path,
    model_dir: Path,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss
    load_started = time.perf_counter()
    reader, artifacts = build_reader(variant, source_dir, model_dir)
    measured_load_seconds = time.perf_counter() - load_started
    loaded_rss = process.memory_info().rss
    peak_rss = loaded_rss
    measurements: list[dict[str, Any]] = []

    for length in LENGTHS:
        text = input_for_length(length)
        for _ in range(warmup):
            reader.infer(text)

        timings: list[float] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(iterations):
                started = time.perf_counter_ns()
                reader.infer(text)
                timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
                peak_rss = max(peak_rss, process.memory_info().rss)
        finally:
            if gc_was_enabled:
                gc.enable()

        stats = percentiles(timings)
        stats.update(
            {
                "characters": len(text),
                "iterations": iterations,
                "chars_per_second": len(text) / (stats["median_ms"] / 1000.0),
                "minimum_ms": min(timings),
                "maximum_ms": max(timings),
            }
        )
        measurements.append(stats)

    return {
        "variant": variant,
        "load_seconds": measured_load_seconds,
        "reported_reader_startup_seconds": getattr(reader, "startup_seconds", None),
        "rss_baseline_bytes": baseline_rss,
        "rss_loaded_bytes": loaded_rss,
        "rss_load_delta_bytes": max(0, loaded_rss - baseline_rss),
        "rss_peak_bytes": peak_rss,
        "artifact_size_bytes": artifact_size(artifacts),
        "artifact_files": [
            os.path.relpath(path, Path.cwd()).replace("\\", "/")
            for path in artifacts
        ],
        "measurements": measurements,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Yomogi ONNX benchmark",
        "",
        (
            f"CPU, intra-op 1, inter-op 1, warmup {report['warmup']}, "
            f"measurement {report['iterations']}; each variant ran in a fresh process."
        ),
        "",
        "| Variant | Chars | Median ms | p95 ms | p99 ms | chars/s | Load s | Peak RSS MiB | Artifacts MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["variants"]:
        for measurement in variant["measurements"]:
            lines.append(
                "| {variant} | {characters} | {median_ms:.3f} | {p95_ms:.3f} | "
                "{p99_ms:.3f} | {chars_per_second:.1f} | {load_seconds:.3f} | "
                "{rss:.1f} | {size:.1f} |".format(
                    variant=variant["variant"],
                    load_seconds=variant["load_seconds"],
                    rss=variant["rss_peak_bytes"] / (1024 * 1024),
                    size=variant["artifact_size_bytes"] / (1024 * 1024),
                    **measurement,
                )
            )
    lines.extend(
        [
            "",
            "RSS is process RSS, not a model-exclusive allocation. Artifact size includes the files required by that variant.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Yomogi PyTorch and ONNX")
    parser.add_argument("--source-dir", type=Path, default=Path(".work/yomogi-v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("dist"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/benchmark.md"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    model_dir = args.model_dir.resolve()

    if args.worker:
        if not args.variant or args.worker_output is None:
            parser.error("worker mode requires --variant and --worker-output")
        result = run_worker(
            args.variant,
            source_dir,
            model_dir,
            args.warmup,
            args.iterations,
        )
        args.worker_output.write_text(json.dumps(result), encoding="utf-8")
        return

    variants = [
        "pytorch_cpu",
        "encoder_fp32_memory",
        "encoder_fp32_mmap",
        "full_fp32",
    ]
    if (model_dir / "experimental/yomogi_encoder_int8.onnx").exists():
        variants.append("encoder_int8_experimental")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="yomogi-benchmark-") as directory:
        temporary = Path(directory)
        for variant in variants:
            output = temporary / f"{variant}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--variant",
                variant,
                "--worker-output",
                str(output),
                "--source-dir",
                str(source_dir),
                "--model-dir",
                str(model_dir),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ]
            print(f"Benchmarking {variant}...", flush=True)
            subprocess.run(command, check=True)
            results.append(json.loads(output.read_text(encoding="utf-8")))

    report = {
        "conditions": {
            "device": "CPU",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "gc_disabled_during_measurement": True,
            "process_isolation": True,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "lengths": list(LENGTHS),
        "python": sys.version,
        "platform": sys.platform,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
