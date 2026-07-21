# TTS Media Server

Discordのボイスチャンネルへ音声を送信するためのAPIサーバー

[TTS Client](https://github.com/haruyq/tts-client)から操作できます。

## 音声プラグイン

プラグインは `application.toml` の `[plugins]` で有効化します。設定例は
`application.example.toml` を参照してください。モデルは初回の音声合成時に
ダウンロード・ロードされます。

### MeloTTS ZH

中国語（英語混在対応）の `ZH` 話者を追加します。

```console
uv sync --extra melotts-zh
```

```toml
[plugins]
melotts_zh = { enabled = true, device = "auto", default_speed = 1.0 }
```

リクエストの `options` では `speed`、`sdp_ratio`、`noise_scale`、
`noise_scale_w` を指定できます。

### Kokoro-82M

Kokoro-82M の54話者を利用できます。`language` は `a`（米語）、`b`（英語）、
`e`（スペイン語）、`f`（フランス語）、`h`（ヒンディー語）、`i`（イタリア語）、
`j`（日本語）、`p`（ブラジルポルトガル語）、`z`（中国語）から選択します。

```console
uv sync --extra kokoro-82m
```

```toml
[plugins]
kokoro_82m = { enabled = true, language = "a", device = "auto", default_speed = 1.0 }
```

リクエストの `options` では `speed` を指定できます。英語の未知語フォールバックと
一部の言語では、OS 側に `espeak-ng` も必要です。

両方を利用する場合はまとめて同期できます。

```console
uv sync --extra melotts-zh --extra kokoro-82m
```
