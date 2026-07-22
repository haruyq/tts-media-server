from __future__ import annotations

from pathlib import Path
import threading

from .runtime import YomogiOnnx


_reader_lock = threading.Lock()
_readers: dict[tuple[str, int, int], YomogiOnnx] = {}


def get_shared_reader(
    model_dir: str | Path,
    *,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
) -> YomogiOnnx:
    """Return one process-wide reader for an equivalent model configuration."""
    resolved = str(Path(model_dir).expanduser().resolve())
    key = (resolved, intra_op_threads, inter_op_threads)
    with _reader_lock:
        reader = _readers.get(key)
        if reader is None:
            reader = YomogiOnnx(
                resolved,
                intra_op_threads=intra_op_threads,
                inter_op_threads=inter_op_threads,
            )
            _readers[key] = reader
        return reader
