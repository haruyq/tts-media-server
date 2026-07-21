from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from reference_model import TorchYomogiReference
from yomogi_onnx.runtime import YomogiOnnx
from yomogi_onnx.types import YomogiResult


def result_signature(result: YomogiResult) -> dict[str, Any]:
    return {
        "normalized_text": result.normalized_text,
        "dict_ids": [token.dict_id for token in result.tokens],
        "surfaces": [token.surface for token in result.tokens],
        "reads": [token.read for token in result.tokens],
        "prons": [token.pron for token in result.tokens],
        "read": result.read,
        "pron": result.pron,
        "unknown_spans": [asdict(value) for value in result.unknown_spans],
    }


def load_curated(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generated_sentences(
    dictionary_path: Path,
    *,
    count: int,
    seed: int = 3135,
) -> list[str]:
    surfaces: list[str] = []
    with dictionary_path.open(encoding="utf-8") as file:
        for line in file:
            row = line.rstrip("\n").split("\t")
            surface = row[1]
            if 1 <= len(surface) <= 8 and not surface.isspace():
                surfaces.append(surface)

    generator = random.Random(seed)
    results: list[str] = []
    endings = ("。", "！", "？")
    for _ in range(count):
        part_count = generator.randint(2, 8)
        text = "".join(generator.choice(surfaces) for _ in range(part_count))
        if len(text) > 120:
            text = text[:120]
        results.append(text + generator.choice(endings))
    return results


def _logit_differences(
    reference_trace: list[dict[str, Any]],
    actual_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    actual_by_position = {value["position"]: value for value in actual_trace}
    for reference in reference_trace:
        actual = actual_by_position.get(reference["position"])
        item: dict[str, Any] = {
            "position": reference["position"],
            "reference_candidate_ids": reference["candidate_ids"],
            "actual_candidate_ids": None if actual is None else actual["candidate_ids"],
            "reference_selected_id": reference["selected_id"],
            "actual_selected_id": None if actual is None else actual["selected_id"],
        }
        if actual is not None and reference["candidate_ids"] == actual["candidate_ids"]:
            reference_logits = np.asarray(reference["logits"], dtype=np.float64)
            actual_logits = np.asarray(actual["logits"], dtype=np.float64)
            delta = np.abs(reference_logits - actual_logits)
            item["max_abs_logit_difference"] = float(delta.max(initial=0.0))
        differences.append(item)
    return differences


def compare_reader(
    name: str,
    reader: YomogiOnnx,
    reference: TorchYomogiReference,
    cases: Iterable[tuple[str, str]],
    failure_file,
) -> dict[str, Any]:
    total = 0
    exact = 0
    dict_id_exact = 0
    read_exact = 0
    pron_exact = 0
    curated_total = 0
    curated_exact = 0
    for category, text in cases:
        total += 1
        if category == "curated":
            curated_total += 1
        expected, expected_trace = reference.infer_debug(text)
        actual, actual_trace = reader.debug_trace(text)
        expected_signature = result_signature(expected)
        actual_signature = result_signature(actual)
        matches = expected_signature == actual_signature
        if matches:
            exact += 1
            if category == "curated":
                curated_exact += 1
        if expected_signature["dict_ids"] == actual_signature["dict_ids"]:
            dict_id_exact += 1
        if expected.read == actual.read:
            read_exact += 1
        if expected.pron == actual.pron:
            pron_exact += 1
        if not matches:
            failure = {
                "reader": name,
                "category": category,
                "input": text,
                "pytorch": expected_signature,
                "onnx": actual_signature,
                "candidate_logit_differences": _logit_differences(
                    expected_trace,
                    actual_trace,
                ),
            }
            failure_file.write(json.dumps(failure, ensure_ascii=False) + "\n")

    return {
        "reader": name,
        "total": total,
        "exact": exact,
        "exact_rate": exact / total if total else 1.0,
        "dict_id_exact": dict_id_exact,
        "dict_id_exact_rate": dict_id_exact / total if total else 1.0,
        "read_exact": read_exact,
        "read_exact_rate": read_exact / total if total else 1.0,
        "pron_exact": pron_exact,
        "pron_exact_rate": pron_exact / total if total else 1.0,
        "curated_total": curated_total,
        "curated_exact": curated_exact,
        "curated_exact_rate": curated_exact / curated_total if curated_total else 1.0,
        "passed_fp32_requirement": exact == total and curated_exact == curated_total,
    }


def run_comparison(
    source_dir: Path,
    model_dir: Path,
    *,
    random_count: int = 1000,
    include_full: bool = True,
    include_int8: bool = False,
    failure_path: Path,
) -> dict[str, Any]:
    curated = load_curated(Path(__file__).parent / "tests" / "ambiguous_sentences.txt")
    random_cases = generated_sentences(
        source_dir / "model" / "dictionary.tsv",
        count=random_count,
    )
    cases = [("curated", text) for text in curated]
    cases.extend(("generated", text) for text in random_cases)

    reference = TorchYomogiReference(source_dir)
    readers: list[tuple[str, YomogiOnnx]] = [
        ("encoder_fp32", YomogiOnnx(str(model_dir))),
    ]
    if include_full and (model_dir / "yomogi_full_fp32.onnx").exists():
        readers.append(
            (
                "full_fp32",
                YomogiOnnx(
                    str(model_dir),
                    model_filename="yomogi_full_fp32.onnx",
                    full_model=True,
                ),
            )
        )
    if include_int8 and (model_dir / "yomogi_encoder_int8.onnx").exists():
        readers.append(
            (
                "encoder_int8",
                YomogiOnnx(
                    str(model_dir),
                    model_filename="yomogi_encoder_int8.onnx",
                ),
            )
        )

    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", encoding="utf-8") as failure_file:
        summaries = [
            compare_reader(name, reader, reference, cases, failure_file)
            for name, reader in readers
        ]
    if failure_path.stat().st_size == 0:
        failure_path.unlink()

    return {
        "source_revision": "3135d1274edf66099fbced229b0048b08e98dd70",
        "curated_cases": len(curated),
        "generated_cases": len(random_cases),
        "seed": 3135,
        "readers": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX outputs")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("dist"))
    parser.add_argument("--random-count", type=int, default=1000)
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--no-full", action="store_true")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=Path("reports/equivalence_failures.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/equivalence_fp32.json"),
    )
    args = parser.parse_args()
    summary = run_comparison(
        args.source_dir.resolve(),
        args.model_dir.resolve(),
        random_count=args.random_count,
        include_full=not args.no_full,
        include_int8=args.include_int8,
        failure_path=args.failure_path.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp32 = [value for value in summary["readers"] if value["reader"].endswith("fp32")]
    if not all(value["passed_fp32_requirement"] for value in fp32):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
