# TTS Media Server

TTS音声ファイルの生成とDiscordのボイスチャンネルへの送信を提供するAPIサーバー

[TTS Client](https://github.com/haruyq/tts-client)から操作できます。

## 公式プラグイン

|  プラグイン  |  ローカル生成  |
|     ----     | --- |
| `voicevox`   | ✅️ (VOICEVOX Core) |
| `coeiroink`  | ✅️ (torch, cpu/cuda) |
| `piper_tts`  | ✅️ (onnxruntime, cpu) |
| `kokoro_82m` | ❌️ (API, make it yourself) |
| `aitalked`   | ❌️ (API, [aitalked-server](https://github.com/yanorei32/aitalked-server)) |

これらはリポジトリからコピーして使用できます。

## 読み補正

`processors/reading`は、[Yomogi](https://huggingface.co/spaces/litagin/yomogi-v1.8)による文脈に応じた読み分け、ユーザー辞書及び英単語のカタカナ変換で、合成前の文を補正します。
別プロセス (torch, cpu/cuda) で動作し、`[plugins.processor]`で指定したプラグインにだけ適用されます。

## ライセンス

このプロジェクトは[MIT License](./LICENSE)で公開されています。
