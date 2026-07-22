from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred, full_check=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    shapes: list[list[int]] = []
    for length in (1, 2, 7, 31, 127):
        result = session.run(
            ["hidden_states"],
            {
                "input_ids": np.zeros((length,), dtype=np.int64),
                "surface_vocab_ids": np.zeros((length,), dtype=np.int64),
            },
        )[0]
        if result.shape[0] != length:
            raise RuntimeError(
                f"Dynamic length validation failed: {length} -> {result.shape}"
            )
        if not np.isfinite(result).all():
            raise RuntimeError("Quantized model produced non-finite hidden states")
        shapes.append(list(result.shape))
    return {
        "checker": "passed",
        "shape_inference": "passed",
        "runtime_shapes": shapes,
        "opset": [
            {"domain": value.domain, "version": value.version}
            for value in model.opset_import
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamically quantize the Yomogi encoder for ONNX Runtime"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quantization.json"),
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        op_types_to_quantize=["LSTM", "MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        use_external_data_format=False,
    )
    validation = validate_model(output_path)
    report = {
        "input": {
            "path": args.input.as_posix(),
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256(input_path),
        },
        "output": {
            "path": args.output.as_posix(),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "settings": {
            "method": "onnxruntime.quantization.quantize_dynamic",
            "op_types_to_quantize": ["LSTM", "MatMul", "Gemm"],
            "weight_type": "QInt8",
            "per_channel": False,
            "reduce_range": False,
        },
        "quantizer_warning": {
            "message": (
                "ONNX Runtime recommended quantization pre-processing before "
                "quantize_dynamic. The emitted model still passed checker, shape "
                "inference, and variable-length runtime validation. Accuracy was "
                "evaluated separately and did not meet the production threshold."
            ),
            "disposition": "retained as experimental; not the default runtime",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "validation": validation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
