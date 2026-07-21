# Yomogi ONNX

Yomogi v1.4を、PyTorchを本番環境へ持ち込まずONNX Runtimeで実行するための変換・推論実装です。Hugging Face Space `prj-beatrice/yomogi-v1` のコミット `3135d1274edf66099fbced229b0048b08e98dd70`（短縮形`3135d12`）に固定しています。Discord TTS向けには、Yomogiの日本語文脈読みとVOICEVOX KanalizerのASCII英単語読みを並列に実行するハイブリッドAPIも提供します。

正式な本番構成は方式AのFP32です。文字埋め込み、表層語埋め込み、4層双方向LSTM、出力射影までを`yomogi_encoder_fp32.onnx`へ入れ、辞書候補の採点と元実装どおりの貪欲デコードをNumPy/Pythonで行います。方式Bの単一ONNXも生成・検証済みですが、候補行列を文章ごとに構築するため既定にはしていません。INT8は精度基準を満たさなかったので`dist/experimental/`に分離しています。

## 検証結果

- ONNX checker、shape inference: 方式A FP32、方式B FP32、実験INT8のすべてで合格
- 可変長実推論: 長さ1、2、7、31、127で合格
- curated曖昧語: 22/22完全一致（方式A/B FP32）
- 固定seed辞書表層回帰: 1,000/1,000完全一致（方式A/B FP32）
- 全1,022文の`dict_id`列、既知tokenのsurface/read/pron列、正規化文字列、unknown_spans: 100%完全一致
- 未知部分なし1,015文の最終read/pron: 100%完全一致
- 未知部分あり7文の元位置・文字列保持: 7/7、方式A/Bの全runtime出力: 1,022/1,022完全一致
- pytest: 24 passed（うち一致テストが上記1,022文×方式A/B）
- PyTorchなしのPython 3.11新規環境: 同期推論・非同期推論とも合格

詳細は`reports/equivalence_fp32.json`、`reports/benchmark.json`、`dist/export_report.json`を参照してください。モデル由来のFP32出力に不一致はありません。旧runtimeが削除していた未知部分を含む7文だけは、意図した新仕様として最終read/pronが元PyTorchラッパーと異なり、未知文字を保持します。INT8では60件の不一致があり、`reports/equivalence_int8_failures.jsonl`に入力、両出力、候補ID、候補logit差を保存しています。

## 推論フロー

方式Aは次の順序です。

1. 元`app.py`と同じUnicode正規化・ASCII全角化
2. `input_ids`と`surface_vocab_ids`を生成（未知文字IDは0）
3. Trieから各位置の候補を元と同じ順番で列挙
4. ONNX Runtimeで`hidden_states [T, D]`を一度生成
5. 選択位置の候補に対応する合成済みweight/biasだけを参照
6. `candidate_weight @ hidden + candidate_bias`をNumPyで計算
7. `argmax`し、選択した表層長だけ進む元実装と同じ貪欲デコード
8. surface、read、pron、dict_idと未知区間を返す

`materialize_output_parameters()`の結果は`candidate_weight_fp32.npy`と`candidate_bias_fp32.npy`へ保存しています。辞書処理をPython側へ残すことで、34万候補を毎位置すべて採点せず、Trieで得た少数候補だけを評価できます。また、候補順序や未知文字の扱いを元実装どおり明示的に維持できます。

方式Bの`yomogi_full_fp32.onnx`は`input_ids [T]`、`surface_vocab_ids [T]`、`candidate_ids [T,C]`、`candidate_mask [T,C]`を受け取り、候補weight/biasのGather、内積、maskをONNX内で行って`logits [T,C]`を返します。Python側はargmaxと貪欲デコードだけを行います。

## インストールと実行

Python 3.11〜3.13を対象にしています。本番の直接依存は`numpy`、`onnxruntime`、公式`kanalizer`です。TorchとGradioは本番依存に含みません。`kanalizer==0.1.1`にはWindows x64/x86、manylinux x64/ARM64、macOS ARM64/x64向けwheelがあります。

```bash
python -m venv .venv
.venv/bin/pip install .
python -m yomogi_onnx "今日は人気の商品です"
python -m yomogi_onnx --json "今日は人気の商品です"
```

Windowsでは`.venv\Scripts\pip.exe`と`.venv\Scripts\python.exe`を使用してください。JSONには`read_katakana`、`pron_katakana`、`pron_hiragana`、TTS既定値の`tts_text`、各tokenの`dict_id`、未知区間、入力順の`segments`が含まれます。各segmentのJSONにも`tts_text`があります。

```python
from yomogi_onnx import YomogiOnnx

reader = YomogiOnnx("./dist", intra_op_threads=1, inter_op_threads=1)
result = reader.infer("今日は人気の商品です")
print(result.pron_hiragana)  # TTSへ渡す既定値
print([token.dict_id for token in result.tokens])
if result.has_unknown:
    print(result.unknown_spans)
```

空文字は安全に空結果を返します。最大長の既定値は500文字で、`max_length=`から変更できます。超過は黙って切らず`ValueError`です。辞書候補がない部分は`unknown_spans`に記録されるだけでなく、元の入力文字と位置を保った`YomogiSegment(is_unknown=True)`として`read`、`pron`、`pron_hiragana`、`tts_text`へ復元されます。連続する未知文字は1つのsegmentに結合されます。`tokens`は後方互換のため、従来どおり辞書で認識された既知語だけです。

### 未知部分を別言語TTSへ振り分ける

Yomogiは候補のない未知部分そのものの読み方を推定しません。未知segmentの`tts_text`は元文字列、既知segmentの`tts_text`はpronをひらがな化した文字列です。そのため、次のように日本語TTSと英語・中国語などの自動言語TTSを分けられます。

```python
for segment in result.segments:
    if segment.is_unknown:
        await synthesize_auto_language(segment.text)
    else:
        await synthesize_japanese(segment.tts_text)
```

未知語を正しく発音したい場合は、別言語TTS、文字読み、ユーザー辞書のいずれかへフォールバックしてください。固定辞書が`Xqz`、数字、記号、いくつかの絵文字を候補として持つ場合、それらは未知ではなく従来どおりYomogiのdict_idと読みが使われます。候補のある文字を強制的に原文通過させる処理は、既知語の推論結果を変えるためこのruntime修正には含めていません。

## Discord.pyへの組み込み

`YomogiOnnx`はBot起動時に1回だけ作ってください。セッション、辞書Trie、候補パラメータはコンストラクタで1回ロードされ、推論ごとに作り直しません。標準設定では1つのlockで推論を直列化するスレッドセーフ設計です。複数guildから同時に呼ばれてもCPU 1スレッドセッションを過剰並列化しません。並列性が必要なら、CPUコア数とメモリを見てreaderを複数用意するか、実測後に`serialize_inference=False`を選んでください。

```python
import discord

from yomogi_onnx import YomogiOnnx, infer_async
from yomogi_onnx.preprocess import preprocess_discord

reader = YomogiOnnx("./dist")  # on_ready/messageごとには作らない

async def text_for_tts(message: discord.Message) -> str:
    prepared = preprocess_discord(
        message.clean_content,
        custom_readings={
            "VRChat": "ぶいあーるちゃっと",
            "Minecraft": "まいんくらふと",
            "RTX5090": "あーるてぃーえっすごーまるきゅーまる",
        },
        max_length=500,
    )
    result = await infer_async(reader, prepared.text)
    return result.pron_hiragana
```

`infer_async()`は`await asyncio.to_thread(reader.infer, text)`の薄いラッパーです。同期コードでは`reader.infer()`を直接呼べます。

`preprocess_discord()`は`message.clean_content`を対象に、URL、カスタム/Unicode絵文字、変換後のメンション表示名、チャンネル名、ロール名、コードブロック、連続記号、連続改行、最大長を処理します。ユーザー辞書はYomogiより前にTrieの最長一致で適用し、非一致部分を一文字ずつ保持するため周辺の単語境界を破壊しません。切り詰めの有無は`DiscordPreprocessResult.truncated`で確認できます。

Discord前処理とユーザー辞書はYomogiより先に適用することを推奨します。前処理後にも候補のない絵文字や中国語などが残った場合は、未知segmentとして削除せず返します。

## Yomogi + Kanalizerハイブリッド読み

`convert_hybrid()`と`convert_hybrid_async()`は、次の優先順位で1本の`tts_text`を合成します。

1. Discord前処理
2. カスタム読み辞書（Trieによる最長一致）
3. ASCII英単語の抽出
4. 全文を渡したYomogiと、重複排除した全英単語を渡したKanalizerの並列実行
5. `prepared_text`上のstart/end offsetによる合成
6. TTS用ひらがな文字列の返却

カスタム辞書の範囲は英単語抽出から除外するため、Kanalizerへ渡りません。組み込み規則には`VRChat`、`RTX5090`、`GPU`、`CPU`、`C++`、`C#`、`.NET`があります。呼び出し時の辞書は同じ表層の組み込み規則を上書きできます。英単語は`[A-Za-z]+(?:['’-][A-Za-z]+)*`で抽出し、Kanalizerが受け付ける小文字ASCIIへ正規化します。`don't`は`dont`、`real-time`は`realtime`として変換します。数字混在の型番全体を無条件にKanalizerへ渡しません。

Kanalizer変換はメッセージ内で重複排除し、プロセス内LRUキャッシュ（最大8,192語）を共有します。入力拒否、変換未完了、空結果では元の英単語をその位置に保持します。予期しない例外もログへ残して原文へフォールバックします。Yomogi候補のない中国語・絵文字・記号も削除しません。`HybridReadingResult.segments`の`source`は`custom`、`yomogi`、`kanalizer`、`unknown`のいずれかです。

同期APIも2ワーカーでYomogiとKanalizerを並列実行します。非同期APIは1メッセージにつきYomogi用とKanalizer全英単語用の`asyncio.to_thread`を1個ずつ作り、`asyncio.gather`します。英単語がなければKanalizerを呼びません。Yomogiへは常に前処理後の全文を渡すため、`人気`、`生`、`一日`、`日本橋`、`上手`などの文脈を分断しません。

```python
import discord

from yomogi_onnx import YomogiOnnx, convert_hybrid_async

reader = YomogiOnnx("./dist")  # Bot起動時に一度だけロード

async def text_for_tts(message: discord.Message) -> str:
    result = await convert_hybrid_async(
        reader,
        message.clean_content,
        custom_readings={
            "VRChat": "ぶいあーるちゃっと",
            "RTX5090": "あーるてぃーえっくすごーまるきゅーまる",
        },
    )
    return result.tts_text
```

たとえば`今日はMinecraftで遊ぶ`は`きょーわまいんくらふとであそぶ`になります。カスタム辞書を先に適用し、Minecraft部分だけをKanalizer、残りの全文文脈をYomogiが担当します。

## 変換

変換時だけPyTorch、onnx、onnxscriptが必要です。元SpaceはGit LFSを含めて取得し、必ず固定コミットをcheckoutしてください。

```bash
git clone https://huggingface.co/spaces/prj-beatrice/yomogi-v1 .work/yomogi-v1
git -C .work/yomogi-v1 checkout 3135d12
git -C .work/yomogi-v1 lfs pull
pip install ".[export]"
python export_onnx.py --source-dir .work/yomogi-v1 --output-dir dist
```

取得した指定8ファイルのbytesとSHA-256は`SOURCE_MANIFEST.json`と`logs/source_files.jsonl`へ記録しています。モデル寸法は`model_meta.json`から読み取り、変換コードへ固定値として埋め込んでいません。

最初に`torch.onnx.export(..., dynamo=True, opset_version=18)`を試しましたが、生成物は長さ4へ固定されたReshapeを含み、長さ1のORT実推論に失敗しました。そのため成功扱いにせず、`dynamo=False`、`dynamic_axes`、opset 18へfallbackしました。fallbackモデルはchecker、shape inference、可変長ORT実推論を通しています。LSTMの可変batch警告はbatch size 1固定という元実装の条件で扱い、可変なのは文章長Tです。全警告と失敗理由は`dist/export_report.json`に保存しています。モデルは外部dataを必要としない通常のONNXファイルで、Netronから直接開けます。

## 量子化

FP32完全一致後にだけ、LSTM、MatMul、Gemmを対象としてORTの動的QInt8量子化を実行しました。Embedding/Gatherは対象外です。

```bash
python quantize_onnx.py \
  --input dist/yomogi_encoder_fp32.onnx \
  --output dist/experimental/yomogi_encoder_int8.onnx
```

INT8は曖昧語22/22（100%）でしたが、全1,022文ではdict_id完全一致962/1,022（94.13%）、read一致968/1,022（94.72%）、pron一致972/1,022（95.11%）でした。99.9%条件を満たさないため実験版であり、既定にしてはいけません。具体的な失敗は`reports/equivalence_int8_failures.jsonl`にあります。

## テスト

```bash
python compare_outputs.py \
  --source-dir .work/yomogi-v1 \
  --model-dir dist \
  --random-count 1000
pytest
```

`tests/ambiguous_sentences.txt`の22文に加え、辞書表層を固定seed 3135で組み合わせた1,000文を比較します。全件でdict_idと既知tokenを比較し、未知部分のない文は最終read/pronも比較します。未知部分を含む文は元入力のlossless復元を確認し、方式A/Bのsegmentsを含むruntime出力を直接比較します。不一致時は入力、PyTorch/ONNX結果、候補ID、候補logit差をJSONLへ出します。候補の並び、`ordered_candidates()`、貪欲デコードの進行方法は変更していません。

## ベンチマーク

```bash
python benchmark.py
python benchmark_hybrid.py --warmup 5 --iterations 30
```

CPU、intra-op 1、inter-op 1、各方式を新規プロセス、warmup 20、本測定200、測定中GC停止の結果です。数値はこのWindowsホストでの実測で、負荷やCPUにより変動します。

| 方式 | 文字数 | median ms | p95 ms | p99 ms | chars/s | load s | peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CPU | 10 | 29.678 | 35.101 | 37.392 | 336.9 | 4.509 | 818.6 |
| 方式A FP32 memory | 10 | 2.261 | 2.809 | 3.110 | 4422.3 | 1.839 | 436.8 |
| 方式A FP32 mmap | 10 | 2.255 | 2.789 | 2.987 | 4434.6 | 1.868 | 394.2 |
| 方式B FP32 | 10 | 2.240 | 2.972 | 5.750 | 4463.9 | 1.913 | 436.6 |
| PyTorch CPU | 30 | 53.216 | 57.246 | 60.447 | 563.7 | 4.509 | 818.6 |
| 方式A FP32 memory | 30 | 38.947 | 43.939 | 45.662 | 770.3 | 1.839 | 436.8 |
| 方式A FP32 mmap | 30 | 40.005 | 43.310 | 44.560 | 749.9 | 1.868 | 394.2 |
| 方式B FP32 | 30 | 39.611 | 43.619 | 114.548 | 757.4 | 1.913 | 436.6 |
| PyTorch CPU | 100 | 157.502 | 172.572 | 247.434 | 634.9 | 4.509 | 818.6 |
| 方式A FP32 memory | 100 | 131.098 | 136.107 | 137.599 | 762.8 | 1.839 | 436.8 |
| 方式A FP32 mmap | 100 | 133.669 | 138.093 | 140.739 | 748.1 | 1.868 | 394.2 |
| 方式B FP32 | 100 | 131.170 | 135.464 | 138.965 | 762.4 | 1.913 | 436.6 |
| PyTorch CPU | 300 | 408.705 | 422.178 | 439.300 | 734.0 | 4.509 | 818.6 |
| 方式A FP32 memory | 300 | 389.614 | 403.607 | 518.560 | 770.0 | 1.839 | 436.8 |
| 方式A FP32 mmap | 300 | 389.281 | 402.940 | 406.182 | 770.7 | 1.868 | 394.2 |
| 方式B FP32 | 300 | 386.917 | 399.771 | 408.444 | 775.4 | 1.913 | 436.6 |

全行と実験INT8は`reports/benchmark.md`および機械可読な`reports/benchmark.json`にあります。通常ロードは短文の小候補参照で安定し、mmapは長文速度をほぼ維持しながらRSSを約42.6 MiB減らしました。既定は通常ロードで、メモリ制約がある環境では`parameter_loading="mmap"`を選べます。

ハイブリッド版は日本語10文字、英単語1個、英単語3個、同一英単語反復を、同期／非同期、LRUキャッシュwarm／coldで測定します。median、p95、Yomogi時間、Kanalizer時間、合計時間を`reports/hybrid_benchmark.md`と`reports/hybrid_benchmark.json`へ保存します。短文Discordメッセージの1回あたりlatencyを対象とした測定です。

## 成果物サイズ

| ファイル | bytes | MiB | SHA-256 |
|---|---:|---:|---|
| `yomogi_encoder_fp32.onnx` | 39,331,536 | 37.51 | `89b86c3c48aa491043276fb2b5729a7f7a85e1d4fc5e3f2dff3325226979b1a0` |
| `candidate_weight_fp32.npy` | 43,810,688 | 41.78 | `f2dd8410b10025b4e8cd42c4c81d13fe8842a4bb21b1e8097c4a8d90218debf4` |
| `candidate_bias_fp32.npy` | 1,369,208 | 1.31 | `1f0e5348806e8adc42a8e7c8c838f78db8f19c63ba5f70a7178c62fb03d0ffed` |
| `yomogi_full_fp32.onnx` | 84,515,266 | 80.60 | `53b2552da4c0d954d626e2f5b3d841087edb4c474c61fe6617d2c33f9f935565` |
| `experimental/yomogi_encoder_int8.onnx` | 23,162,872 | 22.09 | `a91cf47eb4975cdb266a131977f66d3c9edbc066f7ffc0e70d1127f92ea9257f` |

## 既知の制限

- batch sizeは元実装と同じ1です。文章長Tだけが可変です。
- 最大長は既定500文字です。Discord前処理は切り詰めを通知し、直接runtimeは超過を例外にします。
- 未知文字IDは元実装どおり0です。辞書で覆えない部分は読みを推定せず、元文字列のまま最終read/pron/tts_textと`segments`へ保持します。
- Unicode絵文字の判定は実用的なコード範囲ベースであり、新しいEmoji仕様の全シーケンスを意味解析するものではありません。
- INT8は精度不合格です。高速でも本番既定にはしません。
- ベンチマークは単一ホストの測定です。配備先CPUで再測定してください。

## ライセンス

Yomogi v1.4のMIT表記を`LICENSE`に維持しています。辞書はpyopenjtalk-plus/NAIST-jdic/UniDic、AzooKey関連、NEologd、SudachiDict、Mozc UT辞書などの由来を持ち、それぞれの条件が適用されます。元READMEの由来表記は`THIRD_PARTY_NOTICES.md`に保持しています。

VOICEVOX Kanalizer 0.1.1もMIT Licenseです。公式wheelから抽出したライセンス全文を`KANALIZER_LICENSE`、Rust依存を含む第三者通知を`KANALIZER_NOTICE.md`へ保持し、`THIRD_PARTY_NOTICES.md`から参照しています。再配布・配備前に各ライセンス条件も確認してください。
