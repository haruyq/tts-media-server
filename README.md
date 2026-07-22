# TTS Media Server

Discordのボイスチャンネルへ音声を送信するためのAPIサーバー

[TTS Client](https://github.com/haruyq/tts-client)から操作できます。

## 日本語読み前処理

音声合成前の文章は、pluginが任意の`prepare_text()`を持つ場合に全文のまま一度
前処理され、その後で読み上げ単位へ分割されます。VOICEVOXは全話者へYomogi
ONNX＋VOICEVOX Kanalizerを適用します。Kokoro-82Mは既定で`jf_`／`jm_`から
始まる日本語話者だけへ適用し、英語・中国語などの話者やMeloTTS ZHには適用しません。

```toml
[plugins]
voicevox = { enabled = true, base_url = "http://127.0.0.1:50021", japanese_analysis = true }
kokoro_82m = { enabled = false, base_url = "http://127.0.0.1:50101", japanese_analysis = true }
```

Kokoroや将来追加する多言語pluginでは`japanese_speakers`に話者IDを明示できます。
省略時はKokoroの日本語話者prefixを使用します。VOICEVOXで指定した場合は、列挙した
話者だけへ限定できます。

```toml
kokoro_82m = { enabled = true, japanese_analysis = true, japanese_speakers = ["jf_alpha", "jm_kumo"] }
```

Yomogi readerは同じモデル設定についてプロセス内で1個だけ生成され、複数pluginで
共有されます。解析失敗時はログを残して原文をそのままTTSへ渡します。

### 音声処理の追跡ログ

`application.toml`の`server.debug = true`で、各`speech.play`に12桁の`trace_id`を
付けたDEBUGログを出力します。同じIDについて`received`、`prepare.start`、
`prepare.done`、`split`、各`synthesize.start`／`done`、`finished`の順に確認できます。
`received`が既に1語だけなら送信側で分割されており、`received`が全文なのに
`split`の`chunks`が細かければMedia Server側で分割されています。`trace_id`は
`speech.accepted`以降のWebSocketイベントにも含まれます。

## ライセンス

このプロジェクトは[MIT License](./LICENSE)で公開されています。
