import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).parents[1]


def load_engine(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_):
        return self

    def numpy(self):
        return self.values


class MeloTTSZHEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_inside_engine(self):
        module = load_engine(
            "test_melotts_zh_engine",
            ROOT / "engines" / "melotts_zh" / "server.py",
        )
        model = MagicMock()
        model.hps.data.spk2id = {"ZH": 7}
        model.tts_to_file.side_effect = (
            lambda _text, _speaker, output, **_options:
            Path(output).write_bytes(b"melo-engine-wave")
        )
        engine = module.MeloTTSZHEngine(device="cpu")
        engine._model = model

        audio = await engine.synthesize(
            "你好，世界。",
            "ZH",
            {"speed": 1.2},
        )

        self.assertEqual(audio, b"melo-engine-wave")
        self.assertEqual(engine.speakers(), ["ZH"])
        args, options = model.tts_to_file.call_args
        self.assertEqual(args[:2], ("你好，世界。", 7))
        self.assertEqual(options["speed"], 1.2)
        self.assertEqual(
            set(module.app.openapi()["paths"]),
            {"/health", "/speakers", "/synthesize"},
        )


class Kokoro82MEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_inside_engine(self):
        module = load_engine(
            "test_kokoro_82m_engine",
            ROOT / "engines" / "kokoro_82m" / "server.py",
        )
        pipeline = MagicMock(
            return_value=[
                SimpleNamespace(audio=FakeTensor([0.1, 0.2])),
                ("text", "phonemes", FakeTensor([0.3])),
            ],
        )
        soundfile = SimpleNamespace(
            write=MagicMock(
                side_effect=lambda output, *_args, **_options:
                output.write(b"kokoro-engine-wave")
            ),
        )
        torch = SimpleNamespace(
            cat=MagicMock(
                side_effect=lambda chunks: FakeTensor(
                    value
                    for chunk in chunks
                    for value in chunk.values
                )
            ),
        )
        engine = module.Kokoro82MEngine(language="zh", device="cpu")
        engine._pipeline = pipeline

        with patch.dict(
            sys.modules,
            {"soundfile": soundfile, "torch": torch},
        ):
            audio = await engine.synthesize(
                "你好，世界。",
                "zf_xiaoxiao",
                {"speed": 1.1},
            )

        self.assertEqual(audio, b"kokoro-engine-wave")
        self.assertIn("zf_xiaoxiao", engine.speakers())
        pipeline.assert_called_once_with(
            "你好，世界。",
            voice="zf_xiaoxiao",
            speed=1.1,
        )
        soundfile.write.assert_called_once()
        self.assertEqual(soundfile.write.call_args.args[2], 24_000)
        self.assertEqual(
            set(module.app.openapi()["paths"]),
            {"/health", "/speakers", "/synthesize"},
        )
