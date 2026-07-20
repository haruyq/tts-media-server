# TTS Media Server

Discord Voiceへ音声を送信する、TTSプラグイン対応のFastAPIサーバーです。
QueueとDiscord Gateway接続はBot側が持ち、APIとはWebSocketで通信します。

## 必要なもの

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- PATHから実行できるFFmpeg
- VOICEVOX EngineなどのTTSエンジン
- `MESSAGE CONTENT INTENT`を有効にしたDiscord Bot

Discord Botには対象チャンネルの閲覧、メッセージ閲覧、メッセージ送信、
Voice Channelへの接続と発言権限が必要です。

## セットアップ

```powershell
uv sync
Copy-Item application.example.toml application.toml
```

`application.toml`はパスワードを含むためGit管理から除外されています。
コピー後、`server.password`は必ず初期値から変更してください。
16文字以上のランダムな値を推奨します。

```toml
[server]
ip = "127.0.0.1"
port = 8000
debug = false
password = "change-me-before-exposing"

[limits]
max_sessions = 10
max_text_length = 500

[plugins]
voicevox = { enabled = true, base_url = "http://127.0.0.1:50021" }
```

Pluginは設定に名前があり、`enabled`が`true`の場合だけ読み込まれます。
それ以外の値はPlugin固有の設定として渡されます。設定変更後はサーバーを
再起動してください。初期パスワードのままでは起動しません。
外部公開時のHTTPS化はリバースプロキシ側で行ってください。

リポジトリのルートからサーバーを起動します。

```powershell
.\.venv\Scripts\python.exe src\main.py
```

`debug = true`にするとFastAPIのdebugとUvicornのreloadが有効になります。

## 認証

すべての`/api` HTTPリクエストとWebSocket接続に次のヘッダーが必要です。

```text
Authorization: Bearer <application.tomlのserver.password>
```

認証失敗時、HTTPは`401`、WebSocketはclose code `1008`になります。

## HTTP API

- `GET /api/plugins`: 有効なPlugin一覧
- `GET /api/speakers`: Plugin別の話者一覧
- `GET /api/styles`: Plugin、話者別の話し方一覧
- `GET /api/status`: Session使用枠とホスト全体のCPU、RAM使用状況
- `POST /api/sessions/?session_id=...`: Session作成
- `POST /api/sessions/{session_id}/play?path=...`: ローカル音声再生
- `DELETE /api/sessions/{session_id}/playback/current`: 再生停止
- `DELETE /api/sessions/{session_id}`: Session削除

Swagger UIは`/docs`です。右上の`Authorize`へパスワードを入力できます。
WebSocketはOpenAPIに含まれないため、以下を参照してください。

## WebSocket API

接続先は`/api/sessions/{session_id}/ws`です。接続直後に次のイベントを受信します。

```json
{"op": "session.ready", "data": {"session_id": "example"}}
```

Discord Gatewayから得た接続情報を送信します。

```json
{
  "op": "session.create",
  "data": {
    "guild_id": 1,
    "channel_id": 2,
    "user_id": 3,
    "voice_session_id": "voice-session",
    "endpoint": "voice.example.discord.media",
    "token": "voice-token"
  }
}
```

読み上げ要求は次の形式です。

```json
{
  "op": "speech.play",
  "data": {
    "plugin": "voicevox",
    "speaker": "ずんだもん",
    "text": "こんにちは",
    "options": {"style": "あまあま"}
  }
}
```

イベントは次の順序で送信されます。

1. `speech.accepted`: 合成要求を受理
2. `speech.started`: 合成完了後、Discord再生を開始
3. `speech.finished`: 再生完了

失敗時は`speech.failed`、停止時は`speech.stopped`です。`playback.stop`、
`session.close`、`ping`も利用できます。同じSessionで同時に再生できる音声は1件です。

## 簡易Discord Bot

簡易Botは`application.toml`のAPI側IP、port、passwordを参照します。
Queue上限はBot側で100件です。Discord情報は環境変数に設定します。

```powershell
$env:DISCORD_BOT_TOKEN = "..."
$env:DISCORD_GUILD_ID = "..."
$env:DISCORD_VOICE_CHANNEL_ID = "..."
$env:DISCORD_TEXT_CHANNEL_ID = "..."
.\.venv\Scripts\python.exe tests\manual_discord_bot.py
```

任意の環境変数は次のとおりです。

- `TTS_MEDIA_SERVER_URL`
- `TTS_SESSION_ID`（既定値: `manual-test`）
- `TTS_PLUGIN`（既定値: `voicevox`）
- `TTS_SPEAKER`（既定値: `ずんだもん`）
- `TTS_STYLE`
- `TTS_SPEECH_TIMEOUT`（既定値: `120`秒）

BotはAPI切断時に再接続し、Discordのvoice session、token、endpointが変わった場合は
API Sessionを作り直します。未完了の発話は再送するため、切断タイミングによっては
同じ発話を再度読み上げます。Queueが上限に達した場合、新しいメッセージは読み上げません。

## Plugin API

`plugins/{plugin_name}.py`に`plugin`オブジェクトを定義します。

```python
from utils.models import AudioData

class Plugin:
    def configure(self, config) -> None:
        self.endpoint = config["endpoint"]

    async def speakers(self) -> list[str]:
        return ["speaker"]

    async def styles(self) -> dict[str, list[str]]:
        return {"speaker": ["normal"]}

    async def synthesize(self, text, speaker, options) -> AudioData:
        return AudioData(b"...")

plugin = Plugin()
```

`speakers`と`synthesize`は必須、`styles`は任意です。Plugin固有の設定を
使用する場合は同期`configure`を実装します。`configure`には`enabled`を除いた
設定が渡されます。Pluginはサーバープロセスと同じ権限で実行されるため、管理者が
確認した信頼済みコードだけを配置してください。

VOICEVOX Pluginは`base_url`で接続先を変更できます。未指定の場合は
`VOICEVOX_URL`、それも未指定なら`http://127.0.0.1:50021`を使用します。

## テスト

```powershell
Set-Location src
..\.venv\Scripts\python.exe -m unittest discover -s ..\tests -p "test_*.py"
```
