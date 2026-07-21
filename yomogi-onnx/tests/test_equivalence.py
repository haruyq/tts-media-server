import json
import os
from pathlib import Path

from compare_outputs import run_comparison


def test_fp32_matches_fixed_pytorch_model(tmp_path: Path) -> None:
    source_dir = Path(
        os.environ.get("YOMOGI_SOURCE_DIR", ".work/yomogi-v1")
    ).resolve()
    model_dir = Path(os.environ.get("YOMOGI_MODEL_DIR", "dist")).resolve()
    random_count = int(os.environ.get("YOMOGI_RANDOM_CASES", "1000"))
    if not (source_dir / "model/model.pt").exists():
        raise AssertionError("Set YOMOGI_SOURCE_DIR to fixed revision 3135d12")
    summary = run_comparison(
        source_dir,
        model_dir,
        random_count=random_count,
        include_full=True,
        include_int8=False,
        failure_path=tmp_path / "equivalence_failures.jsonl",
    )
    failures = [
        value
        for value in summary["readers"]
        if not value["passed_fp32_requirement"]
    ]
    assert not failures, json.dumps(failures, ensure_ascii=False, indent=2)
    cross_check = summary["onnx_cross_check"]
    assert cross_check is not None
    assert cross_check["exact"] == cross_check["total"] == 1022
