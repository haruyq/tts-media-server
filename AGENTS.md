# TTS Media Server

TTS Media Serverは、主にDiscordのTTS Bot向けに開発された音声配信APIです。
このAPIの責務は、外部エンジンによるTTS生成及び生成された音声の配信です。

## プラグイン開発ガイド

プラグインはAPI本体とは別のPythonプロセスで動作します。
外部TTSエンジンとの通信だけでなく、プラグインプロセス内での音声生成も可能です。
API本体はプラグインの依存関係を読み込まず、プロセスの起動、監視及び音声データの受け渡しだけを行います。

公式のサンプルは`plugins/voicevox`及び`plugins/kokoro_82m`に含まれます。

プラグインには、`synthesize`関数と`speakers`関数が必須で、`configure`関数及び`styles`関数は任意で追加できます。
TTSバックエンドに複数の喋り方(スタイル)が登録されている場合は、`styles`関数を定義できます。

### プラグインの構成

プラグインは`plugins/<プラグイン名>`に配置します。
各ディレクトリには`plugin.toml`とエントリーポイントが必要です。

```text
plugins/
└── example/
    ├── plugin.toml
    ├── plugin.py
    └── resources/
```

`plugin.toml`にはAPIバージョン、エントリーポイント及び依存関係を記述します。
依存関係はPEP 508形式で、同じruntimeを使用する有効なプラグインの分をuvがまとめてインストールします。

```toml
api_version = 1
entrypoint = "plugin.py"
dependencies = [
    "aiohttp>=3.14.1,<4",
]
```

プラグインへ同梱したwheelは`"./vendor/example.whl"`のように指定できます。
相対パスはプラグインのディレクトリを基準に解決されます。

エントリーポイントのモジュール直下に`plugin`変数を公開してください。

必須及び任意のインターフェースは次の通りです。

```python
from typing import Any

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
    ) -> bytes:
        ...

plugin = ExamplePlugin()
```

`configure`は同期関数、それ以外は非同期関数として実装してください。
`speakers`は利用可能な話者名、`styles`は話者名をキー、スタイル名の一覧を値とする辞書を返します。
`synthesize`はWAV音声のバイト列を返してください。
存在しない話者、スタイル又は不正なオプションは`ValueError`として扱ってください。

### プラグインの設定

プラグインは`application.toml`の`[plugins]`で明示的に有効化します。
設定名はプラグインのディレクトリ名と一致させてください。

```toml
[plugins]
example = { enabled = true, base_url = "http://127.0.0.1:50021" }

[plugins.runtime]
example = "python"
```

`enabled`はAPI本体が処理し、それ以外の値だけが`configure`へ渡されます。
設定項目を持つプラグインは`configure`を定義し、値の型、必須項目及び未知の項目を検証してください。
設定が不正な場合は、起動を継続せず明確な例外を送出してください。

`[plugins.runtime]`の値が同じプラグインはPython環境を共有します。
runtimeはユーザーキャッシュ内へ必要時に作成され、プラグインの依存関係が変更されたときにuvで更新されます。
保存先は`TTS_MEDIA_SERVER_RUNTIME_DIR`で変更できます。

Torchを含む依存関係では、通常のruntime名でもuvがbackendを自動選択します。
明示的に選択する場合は`torch-auto`、`torch-cpu`、`torch-cu128`のように`torch-<uv backend>`を指定してください。
異なるTorch又はCUDA構成が必要なプラグインには、異なるruntime名を指定します。
GPU構成の変更又は同じパスにある同梱wheelの差し替え後は、該当runtimeのディレクトリを削除すると再構築されます。

### 実装及びテストの方針

- 外部TTSバックエンドとの通信には`aiohttp`を優先し、`plugin.toml`の依存関係へ追加してください。
- HTTPリクエストにはタイムアウトを設定し、失敗したレスポンスを正常な音声として扱わないでください。
- 非同期関数内で同期的なネットワーク通信や重い音声生成処理を実行しないでください。
- プラグイン固有の処理はプラグイン内に閉じ込め、共通インターフェースの変更が必要な場合のみAPI本体を変更してください。
- 標準出力はAPI本体との通信に使用するため、ログには標準エラー出力又は`logging`を使用してください。
- プロセス分離はセキュリティ上のサンドボックスではありません。信頼できるプラグインだけを導入してください。
- 新しいプラグインを追加する場合は`application.example.toml`へ設定例を追加してください。
- テストでは実際のTTSバックエンドへ接続できる場合は接続し、テストを行ってください。
- プラグインのみを変更した場合は、まず`uv run python -m unittest tests.test_plugins`を実行してください。
