# TTS Media Server

Discordのボイスチャンネルへ音声を送信するためのAPIサーバー

[TTS Client](https://github.com/haruyq/tts-client)から操作できます。

## Docker音声エンジン

MeloTTS ZH と Kokoro-82M はメディアサーバーのPython環境には入りません。
`plugins/` はHTTPクライアントのみで、PyTorchとモデルは独立したDocker
コンテナ内で動作します。

```text
TTS Media Server -> HTTP plugin -> Docker engine -> PyTorch model
```

両エンジンをビルドして起動します。

```console
docker compose up --build -d
```

初回合成時にモデルをダウンロードします。キャッシュはDockerの名前付きボリュームに
保存されるため、コンテナを作り直しても再利用されます。

| エンジン | ホスト側URL | コンテナ内の実装 |
| --- | --- | --- |
| MeloTTS ZH | `http://127.0.0.1:50100` | `engines/melotts_zh` |
| Kokoro-82M | `http://127.0.0.1:50101` | `engines/kokoro_82m` |

稼働確認:

```console
curl http://127.0.0.1:50100/health
curl http://127.0.0.1:50101/health
```

### プラグイン設定

`application.toml` の `[plugins]` で、起動したエンジンのURLを指定します。

```toml
[plugins]
melotts_zh = { enabled = true, base_url = "http://127.0.0.1:50100" }
kokoro_82m = { enabled = true, base_url = "http://127.0.0.1:50101" }
```

MeloTTS ZH は `ZH` 話者を提供し、リクエストの `options` で `speed`、
`sdp_ratio`、`noise_scale`、`noise_scale_w` を指定できます。

Kokoro-82M は `KOKORO_82M_LANGUAGE` で言語を選択します。値は `a`（米語）、
`b`（英語）、`e`（スペイン語）、`f`（フランス語）、`h`（ヒンディー語）、
`i`（イタリア語）、`j`（日本語）、`p`（ブラジルポルトガル語）、
`z`（中国語）です。`options` では `speed` を指定できます。

環境変数は `docker compose` 実行時に上書きできます。

```console
KOKORO_82M_LANGUAGE=z KOKORO_82M_DEVICE=cpu docker compose up --build -d
```

Windows PowerShellの場合:

```powershell
$env:KOKORO_82M_LANGUAGE = "z"
$env:KOKORO_82M_DEVICE = "cpu"
docker compose up --build -d
```

CUDAを利用する場合は、NVIDIA Container Toolkitを設定したうえでCompose設定に
GPU割り当てを追加し、各エンジンの `*_DEVICE` を `cuda` にします。
