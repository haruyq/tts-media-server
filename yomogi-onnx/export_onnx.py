from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import warnings
from typing import Any, Callable

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort
import torch
from torch import nn

from reference_model import CandidateModel, load_model


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


REQUIRED_FILES = (
    "app.py",
    "model/model.pt",
    "model/model_meta.json",
    "model/dictionary.tsv",
    "model/input_tokens.tsv",
    "model/surface_vocab.tsv",
    "model/read_chars.tsv",
    "README.md",
)


class FullCandidateModel(nn.Module):
    def __init__(
        self,
        encoder: CandidateModel,
        candidate_weight: torch.Tensor,
        candidate_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("candidate_weight", candidate_weight)
        self.register_buffer("candidate_bias", candidate_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        surface_vocab_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, surface_vocab_ids)
        weight = self.candidate_weight[candidate_ids]
        bias = self.candidate_bias[candidate_ids]
        logits = (weight * hidden.unsqueeze(1)).sum(dim=-1) + bias
        return torch.where(candidate_mask, logits, torch.full_like(logits, -1.0e30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest["files"]}
    results: list[dict[str, Any]] = []
    for relative in REQUIRED_FILES:
        path = source_dir / relative
        actual = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        wanted = expected[relative]
        if actual["bytes"] != wanted["bytes"] or actual["sha256"] != wanted["sha256"]:
            raise ValueError(f"Source checksum mismatch: {relative}")
        print(json.dumps(actual, ensure_ascii=False))
        results.append(actual)
    return results


def _export_attempt(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    dynamic_shapes: tuple[dict[int, Any], ...],
    report: dict[str, Any],
    label: str,
    validator: Callable[[], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    attempts: list[tuple[str, dict[str, Any]]] = [
        (
            "dynamo=True, opset=18",
            {
                "dynamo": True,
                "opset_version": 18,
                "dynamic_shapes": dynamic_shapes,
            },
        ),
        (
            "dynamo=False, dynamic_axes, opset=18",
            {
                "dynamo": False,
                "opset_version": 18,
                "dynamic_axes": dynamic_axes,
            },
        ),
        (
            "dynamo=False, dynamic_axes, opset=17",
            {
                "dynamo": False,
                "opset_version": 17,
                "dynamic_axes": dynamic_axes,
            },
        ),
    ]
    report[label] = {"attempts": []}
    for name, options in attempts:
        attempt: dict[str, Any] = {"name": name}
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                torch.onnx.export(
                    model,
                    args,
                    str(output_path),
                    input_names=input_names,
                    output_names=output_names,
                    external_data=False,
                    **options,
                )
            validation = validator()
            attempt["warnings"] = [str(value.message) for value in caught]
            attempt["validation"] = validation
            attempt["seconds"] = time.perf_counter() - started
            attempt["success"] = True
            report[label]["attempts"].append(attempt)
            report[label]["selected"] = name
            return name, validation
        except Exception as exc:
            attempt.update(
                {
                    "success": False,
                    "seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            report[label]["attempts"].append(attempt)
    raise RuntimeError(f"All ONNX export attempts failed for {label}")


def validate_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, path, save_as_external_data=False)
    reloaded = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(reloaded, full_check=True)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "ir_version": reloaded.ir_version,
        "opsets": [
            {"domain": value.domain, "version": value.version}
            for value in reloaded.opset_import
        ],
        "external_data": any(
            initializer.data_location == onnx.TensorProto.EXTERNAL
            for initializer in reloaded.graph.initializer
        ),
    }


def validate_encoder_runtime(
    model: CandidateModel,
    path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    results: list[dict[str, Any]] = []
    max_abs = 0.0
    with torch.inference_mode():
        for length in (1, 2, 7, 31, 127):
            input_ids = np.arange(length, dtype=np.int64) % int(meta["input_vocab_size"])
            surface_ids = np.arange(length, dtype=np.int64) % int(meta["surface_vocab_size"])
            expected = model(torch.from_numpy(input_ids), torch.from_numpy(surface_ids)).numpy()
            actual = session.run(
                ["hidden_states"],
                {"input_ids": input_ids, "surface_vocab_ids": surface_ids},
            )[0]
            difference = float(np.max(np.abs(expected - actual)))
            max_abs = max(max_abs, difference)
            results.append(
                {"length": length, "shape": list(actual.shape), "max_abs_error": difference}
            )
    return {"lengths": results, "max_abs_error": max_abs}


def validate_full_runtime(
    model: FullCandidateModel,
    path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    max_abs = 0.0
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for length, count in ((1, 1), (3, 5), (17, 11)):
            input_ids = np.arange(length, dtype=np.int64) % int(meta["input_vocab_size"])
            surface_ids = np.arange(length, dtype=np.int64) % int(meta["surface_vocab_size"])
            candidate_ids = np.arange(length * count, dtype=np.int64).reshape(length, count)
            candidate_ids %= int(meta["dictionary_size"])
            mask = np.ones((length, count), dtype=np.bool_)
            if count > 1:
                mask[:, -1] = False
            expected = model(
                torch.from_numpy(input_ids),
                torch.from_numpy(surface_ids),
                torch.from_numpy(candidate_ids),
                torch.from_numpy(mask),
            ).numpy()
            actual = session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "surface_vocab_ids": surface_ids,
                    "candidate_ids": candidate_ids,
                    "candidate_mask": mask,
                },
            )[0]
            finite = mask
            difference = float(np.max(np.abs(expected[finite] - actual[finite])))
            max_abs = max(max_abs, difference)
            results.append(
                {"length": length, "candidate_count": count, "max_abs_error": difference}
            )
    return {"cases": results, "max_abs_error": max_abs}


def export(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source": verify_source(source_dir, project_dir / "SOURCE_MANIFEST.json"),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    model, meta = load_model(source_dir)
    if model._materialized_output_weight is None or model._materialized_output_bias is None:
        raise RuntimeError("Materialized candidate parameters are missing")

    candidate_weight = model._materialized_output_weight.cpu().numpy().astype(np.float32, copy=True)
    candidate_bias = model._materialized_output_bias.cpu().numpy().astype(np.float32, copy=True)
    np.save(output_dir / "candidate_weight_fp32.npy", candidate_weight, allow_pickle=False)
    np.save(output_dir / "candidate_bias_fp32.npy", candidate_bias, allow_pickle=False)
    report["candidate_parameters"] = {
        "weight_shape": list(candidate_weight.shape),
        "bias_shape": list(candidate_bias.shape),
        "weight_sha256": sha256(output_dir / "candidate_weight_fp32.npy"),
        "bias_sha256": sha256(output_dir / "candidate_bias_fp32.npy"),
    }

    for name in (
        "dictionary.tsv",
        "input_tokens.tsv",
        "surface_vocab.tsv",
        "read_chars.tsv",
        "model_meta.json",
    ):
        shutil.copy2(source_dir / "model" / name, output_dir / name)
    shutil.copy2(project_dir / "SOURCE_MANIFEST.json", output_dir / "SOURCE_MANIFEST.json")
    shutil.copy2(project_dir / "LICENSE", output_dir / "LICENSE")
    shutil.copy2(project_dir / "THIRD_PARTY_NOTICES.md", output_dir / "THIRD_PARTY_NOTICES.md")

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    surface_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    sequence = torch.export.Dim("sequence_length", min=1, max=500)
    encoder_path = output_dir / "yomogi_encoder_fp32.onnx"
    _, encoder_validation = _export_attempt(
        model,
        (input_ids, surface_ids),
        encoder_path,
        input_names=["input_ids", "surface_vocab_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "sequence_length"},
            "surface_vocab_ids": {0: "sequence_length"},
            "hidden_states": {0: "sequence_length"},
        },
        dynamic_shapes=({0: sequence}, {0: sequence}),
        report=report,
        label="encoder_export",
        validator=lambda: {
            "model": validate_onnx(encoder_path),
            "runtime": validate_encoder_runtime(model, encoder_path, meta),
        },
    )
    report["encoder_model"] = encoder_validation["model"]
    report["encoder_runtime_validation"] = encoder_validation["runtime"]

    full_model = FullCandidateModel(
        model,
        model._materialized_output_weight,
        model._materialized_output_bias,
    ).eval()
    candidate_ids = torch.tensor(
        [[1, 2, 0], [3, 4, 5], [6, 0, 0], [7, 8, 9]],
        dtype=torch.long,
    )
    candidate_mask = candidate_ids != 0
    candidate_count = torch.export.Dim("candidate_count", min=1)
    full_path = output_dir / "yomogi_full_fp32.onnx"
    try:
        _, full_validation = _export_attempt(
            full_model,
            (input_ids, surface_ids, candidate_ids, candidate_mask),
            full_path,
            input_names=[
                "input_ids",
                "surface_vocab_ids",
                "candidate_ids",
                "candidate_mask",
            ],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "sequence_length"},
                "surface_vocab_ids": {0: "sequence_length"},
                "candidate_ids": {0: "sequence_length", 1: "candidate_count"},
                "candidate_mask": {0: "sequence_length", 1: "candidate_count"},
                "logits": {0: "sequence_length", 1: "candidate_count"},
            },
            dynamic_shapes=(
                {0: sequence},
                {0: sequence},
                {0: sequence, 1: candidate_count},
                {0: sequence, 1: candidate_count},
            ),
            report=report,
            label="full_export",
            validator=lambda: {
                "model": validate_onnx(full_path),
                "runtime": validate_full_runtime(full_model, full_path, meta),
            },
        )
        report["full_model"] = full_validation["model"]
        report["full_runtime_validation"] = full_validation["runtime"]
    except Exception as exc:
        report["full_model_error"] = {"type": type(exc).__name__, "message": str(exc)}
        full_path.unlink(missing_ok=True)

    report_path = output_dir / "export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Yomogi v1.4 to ONNX")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.source_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
