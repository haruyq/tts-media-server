# TTS Media Server

TTS Media Serverは、主にDiscordのTTS Bot向けに開発された音声配信APIです。
このAPIの責務は、外部エンジンによるTTS生成及び生成された音声の配信です。

## プラグイン開発ガイド

このAPIにおける「プラグイン」とは、外部のTTSエンジンへリクエストを投げ、レスポンスの音声データをAPI本体へ伝達させるものです。
プラグイン単体で音声の生成まで行うことは想定しないため、公式のTTSバックエンドが存在しないモデルの場合は対応するバックエンドを自作する必要があります。これは別のプロジェクトとして開発することを想定しています。
TTSモデルのバックエンドを開発する際は、Docker + Composeの仕様を強く推奨します。

公式のサンプルプラグインは`plugins/voicevox.py`に含まれます。
これを参考にプラグインを開発できます。

プラグインには、`synthesize`関数と`speakers`関数が必須で、`configure`関数及び`styles`関数は任意で追加できます。
TTSバックエンドに複数の喋り方(スタイル)が登録されている場合は、`styles`関数を定義できます。

### プラグインの構成

プラグインは`plugins/<プラグイン名>.py`に1ファイルで実装し、モジュール直下の`plugin`変数でインスタンスを公開してください。
ファイル名の拡張子を除いた部分が、API及び設定で使用するプラグイン名になります。

必須及び任意のインターフェースは次の通りです。

```python
from typing import Any

from utils.models import AudioData

class ExamplePlugin:
    def configure(self, config: dict[str, Any]) -> None:
        ...

    async def speakers(self) -> list[str]:
        ...

    async def styles(self) -> dict[str, list[str]]:
        ...

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        ...

plugin = ExamplePlugin()
```

`configure`は同期関数、それ以外は非同期関数として実装してください。
`speakers`は利用可能な話者名、`styles`は話者名をキー、スタイル名の一覧を値とする辞書を返します。
`synthesize`はバックエンドから受け取った音声のバイト列を`AudioData`に格納して返してください。
存在しない話者、スタイル又は不正なオプションは`ValueError`として扱ってください。

### プラグインの設定

プラグインは`application.toml`の`[plugins]`で明示的に有効化します。
設定名はプラグインのファイル名と一致させてください。

```toml
[plugins]
example = { enabled = true, base_url = "http://127.0.0.1:50021" }
```

`enabled`はAPI本体が処理し、それ以外の値だけが`configure`へ渡されます。
設定項目を持つプラグインは`configure`を定義し、値の型、必須項目及び未知の項目を検証してください。
設定が不正な場合は、起動を継続せず明確な例外を送出してください。

### 実装及びテストの方針

- 外部TTSバックエンドとの通信には、既に依存している`aiohttp`を優先して使用してください。
- HTTPリクエストにはタイムアウトを設定し、失敗したレスポンスを正常な音声として扱わないでください。
- 非同期関数内で同期的なネットワーク通信や重い音声生成処理を実行しないでください。
- プラグイン固有の処理はプラグイン内に閉じ込め、共通インターフェースの変更が必要な場合のみAPI本体を変更してください。
- 新しいプラグインを追加する場合は`application.example.toml`へ設定例を追加してください。
- テストでは実際のTTSバックエンドへ接続できる場合は接続し、テストを行ってください。
- プラグインのみを変更した場合は、まず`uv run python -m unittest tests.test_plugins`を実行してください。

