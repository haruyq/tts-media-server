import asyncio
import ctypes
import io
import logging
import math
import os
import shutil
import subprocess
import time
import wave

from collections import OrderedDict
from pathlib import Path
from typing import Any

Log = logging.getLogger(__name__)
# プラグインプロセスではloggingが設定されないため、情報ログも標準エラー出力へ
# 出力する。API本体がプラグイン名を付けてログへ転送する
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
Log.addHandler(_log_handler)
Log.setLevel(logging.INFO)
Log.propagate = False

PLUGIN_DIR = Path(__file__).resolve().parent

def _silence() -> bytes:
    # VOICEVOXの出力と同じ24kHz、16bitモノラルの0.1秒の無音
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(bytes(4800))

    return buffer.getvalue()

SILENCE = _silence()
DEVICES = {"auto": "AUTO", "cpu": "CPU", "gpu": "GPU"}
DEFAULT_STYLE = "ノーマル"
DEFAULT_OPTIONS = {
    "speed_scale": 1.0,
    "pitch_scale": 0.0,
    "intonation_scale": 1.0,
    "volume_scale": 1.0,
}

class VoicevoxPlugin:
    def __init__(self) -> None:
        self._synthesizer: Any = None
        # 話者名 -> スタイル名 -> (スタイルID, VVMファイル)
        self._speakers: dict[str, dict[str, tuple[int, Path]]] = {}
        # VVMファイル -> [モデルID, 使用回数]
        self._loaded: OrderedDict[Path, list[Any]] = OrderedDict()
        self._max_loaded_models = 3
        self._reload_after = 100

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "core_dir",
            "device",
            "max_loaded_models",
            "reload_after",
        }

        if unknown:
            raise ValueError(
                f"Unknown voicevox config: {', '.join(sorted(unknown))}"
            )

        core_dir = config.get("core_dir", "voicevox_core")

        if not isinstance(core_dir, str) or not core_dir.strip():
            raise ValueError("voicevox.core_dir must be a non-empty string")

        device = config.get("device", "auto")

        if device not in DEVICES:
            raise ValueError('voicevox.device must be "auto", "cpu" or "gpu"')

        max_loaded_models = config.get("max_loaded_models", 3)

        if (
            isinstance(max_loaded_models, bool)
            or not isinstance(max_loaded_models, int)
            or max_loaded_models <= 0
        ):
            raise ValueError(
                "voicevox.max_loaded_models must be a positive integer"
            )

        reload_after = config.get("reload_after", 100)

        if (
            isinstance(reload_after, bool)
            or not isinstance(reload_after, int)
            or reload_after < 0
        ):
            raise ValueError(
                "voicevox.reload_after must be 0 (disabled) or "
                "a positive integer"
            )

        self._max_loaded_models = max_loaded_models
        self._reload_after = reload_after
        path = Path(core_dir).expanduser()

        if not path.is_absolute():
            path = PLUGIN_DIR / path

        from voicevox_core.blocking import (
            Onnxruntime,
            OpenJtalk,
            Synthesizer,
            VoiceModelFile,
        )

        lib_dir = path / "onnxruntime" / "lib"
        # CUDA版は推奨とは異なるバージョンが配布され、ダウンローダーは
        # シンボリックリンクを展開しないため、libvoicevox_onnxruntime.so.1.17.3
        # のようなバージョン付きの名前も探す
        onnxruntime_path = next(
            (
                library
                for library in (
                    lib_dir / Onnxruntime.LIB_RECOMMENDED_VERSIONED_FILENAME,
                    *sorted(lib_dir.glob(
                        f"{Onnxruntime.LIB_RECOMMENDED_UNVERSIONED_FILENAME}*"
                    )),
                )
                if library.is_file()
            ),
            None,
        )
        dict_dirs = sorted((path / "dict").glob("open_jtalk_dic_utf_8-*"))
        vvms = sorted((path / "models" / "vvms").glob("*.vvm"))
        missing = [
            name
            for name, found in (
                (f"{lib_dir}/*onnxruntime*", onnxruntime_path),
                (f"{path / 'dict'}/open_jtalk_dic_utf_8-*", dict_dirs),
                (f"{path / 'models' / 'vvms'}/*.vvm", vvms),
            )
            if not found
        ]

        if missing:
            raise ValueError(
                f"VOICEVOX CORE files not found: {', '.join(missing)} "
                "(run the VOICEVOX CORE downloader with --exclude c-api)"
            )

        acceleration_mode = DEVICES[device]

        if device != "cpu":
            failed = self._preload_libraries(path / "additional_libraries")

            # 一部だけ読み込めた状態でGPUを使うと、合成中にcuDNNが
            # プロセスを終了させるため、GPUを使用しない
            if failed:
                message = f"Unable to load GPU libraries: {', '.join(failed)}"

                if device == "gpu":
                    raise ValueError(message)

                Log.warning(f"{message}; falling back to CPU")
                acceleration_mode = "CPU"

        onnxruntime = Onnxruntime.load_once(filename=str(onnxruntime_path))
        self._synthesizer = Synthesizer(
            onnxruntime,
            OpenJtalk(dict_dirs[-1]),
            acceleration_mode=acceleration_mode,
            # https://github.com/VOICEVOX/voicevox_core/issues/888
            cpu_num_threads=max(os.cpu_count() or 1, 2),
        )

        characters = []

        for vvm in vvms:
            with VoiceModelFile.open(vvm) as model:
                characters.extend((meta, vvm) for meta in model.metas)

        self._speakers = {}

        for meta, vvm in sorted(
            characters,
            key=lambda item: item[0].order or 0,
        ):
            for style in sorted(meta.styles, key=lambda s: s.order or 0):
                if style.type == "talk":
                    self._speakers.setdefault(meta.name, {})[style.name] = (
                        style.id,
                        vvm,
                    )

        if self._synthesizer.is_gpu_mode:
            Log.info(f"Using GPU: {self._gpu_name(onnxruntime)}")
        else:
            devices = onnxruntime.supported_devices()
            Log.info(f"Using CPU (supported devices: {devices})")

        styles = sum(len(styles) for styles in self._speakers.values())
        Log.info(
            f"Found {len(self._speakers)} VOICEVOX speakers "
            f"({styles} styles, {len(vvms)} models)"
        )

    async def speakers(self) -> list[str]:
        return list(self._speakers)

    async def styles(self) -> dict[str, list[str]]:
        return {
            speaker: list(styles)
            for speaker, styles in self._speakers.items()
        }

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        try:
            styles = self._speakers[speaker]
        except KeyError:
            raise ValueError(f"Speaker not found: {speaker}")

        if not text.strip():
            raise ValueError("text must be a non-empty string")

        options = dict(options)
        style = options.pop("style", None)

        if style is None:
            style = DEFAULT_STYLE if DEFAULT_STYLE in styles else next(
                iter(styles)
            )
        elif not isinstance(style, str):
            raise ValueError("style must be a string")

        try:
            style_id, vvm = styles[style]
        except KeyError:
            raise ValueError(
                f"Style not found: speaker={speaker}, style={style}"
            )

        return await asyncio.to_thread(
            self._synthesize,
            text,
            style_id,
            vvm,
            self._options(options),
        )

    def _synthesize(
        self,
        text: str,
        style_id: int,
        vvm: Path,
        options: dict[str, float],
    ) -> bytes:
        from voicevox_core import AnalyzeTextError
        from voicevox_core.blocking import VoiceModelFile

        start = time.perf_counter()

        # ONNX Runtimeのメモリプールは処理した最長の出力に合わせて拡張され、
        # モデルを解放するまで縮小しない。保持するモデル数を制限し、
        # 一定回数使用したモデルも解放して、メモリ使用量を元に戻す
        if vvm in self._loaded:
            self._loaded.move_to_end(vvm)
        else:
            while len(self._loaded) >= self._max_loaded_models:
                self._unload(next(iter(self._loaded)))

            with VoiceModelFile.open(vvm) as model:
                self._synthesizer.load_voice_model(model)
                self._loaded[vvm] = [model.id, 0]

            Log.info(f"Loaded VOICEVOX model {vvm.name}")

        try:
            query = self._synthesizer.create_audio_query(text, style_id)
        except AnalyzeTextError:
            # 「？」等の記号だけの文には読む音素が無い。VOICEVOX ENGINEと同様に
            # エラーにせず無音を返す
            return SILENCE

        for name, value in options.items():
            setattr(query, name, value)

        audio = self._synthesizer.synthesis(query, style_id)
        self._loaded[vvm][1] += 1

        if self._reload_after and self._loaded[vvm][1] >= self._reload_after:
            self._unload(vvm)

        elapsed = (time.perf_counter() - start) * 1000
        Log.debug(
            f"synthesis completed in {elapsed:.2f} ms - "
            f"text length: {len(text)}"
        )
        return audio

    def _unload(self, vvm: Path) -> None:
        model_id, uses = self._loaded.pop(vvm)
        self._synthesizer.unload_voice_model(model_id)
        Log.info(f"Released VOICEVOX model {vvm.name} after {uses} uses")

    @staticmethod
    def _options(options: dict[str, Any]) -> dict[str, float]:
        unknown = set(options) - set(DEFAULT_OPTIONS)

        if unknown:
            raise ValueError(
                f"Unknown voicevox options: {', '.join(sorted(unknown))}"
            )

        result: dict[str, float] = {}

        for name, default in DEFAULT_OPTIONS.items():
            value = options.get(name, default)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

            if name == "speed_scale" and value <= 0:
                raise ValueError("speed_scale must be greater than 0")

            if name in ("volume_scale", "intonation_scale") and value < 0:
                raise ValueError(f"{name} must be 0 or greater")

            result[name] = float(value)

        return result

    @staticmethod
    def _preload_libraries(path: Path) -> list[str]:
        # CUDA版のダウンローダーが配置するCUDA及びcuDNNは、ONNX Runtimeの
        # 検索パスに含まれない。先に読み込んでおくと、同じ名前のライブラリとして
        # 再利用される。依存先より先に読み込めない場合があるため、失敗した
        # ものは読み込めるものが無くなるまで再試行する
        pending = [
            library
            for library in sorted(path.glob("*"))
            if library.suffix == ".dll" or ".so" in library.suffixes
        ]

        while pending:
            failed = []

            for library in pending:
                try:
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    failed.append(library)

            if len(failed) == len(pending):
                break

            pending = failed

        return [library.name for library in pending]

    @staticmethod
    def _gpu_name(onnxruntime: Any) -> str:
        if onnxruntime.supported_devices().cuda and shutil.which("nvidia-smi"):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return f"CUDA, {result.stdout.strip().splitlines()[0]}"

        return str(onnxruntime.supported_devices())

plugin = VoicevoxPlugin()
