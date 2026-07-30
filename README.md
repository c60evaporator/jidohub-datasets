# jidohub-datasets

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォームです。
**jidohub-datasets は「データセット固有の形式 → 標準スキーマ」の変換**を担います。

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストする Web プラットフォーム |
| jidohub-core | 標準スキーマ・Hub クライアント・config パーサ |
| jidohub-agents | Agent をロードして実行する Python API |
| **jidohub-datasets（本リポジトリ）** | Dataset をロードして `Sample` に正規化する |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

本リポジトリは **`jidohub-core` にのみ依存**します（agents / interfaces / web には依存しない）。
また **`torch` に依存しません** — データセットの読み込みに学習フレームワークは不要です。

最初に対応するデータセットは nuScenes です。

## インストール

**ローカル開発では `jidohub-core` を先に editable install する。** core は PyPI 未公開のため、
順序が逆だと pip が PyPI を探しに行って失敗する（隣に `jidohub-core` を clone してある前提）。

```bash
pip install -e ../jidohub-core
pip install -e '.[dev,nuscenes]'
```

PyPI 公開後は次のように入る（`nuscenes` extra は Adapter を使う場合のみ）。

```bash
pip install jidohub-datasets            # 変換ロジック + Pillow デコーダ
pip install 'jidohub-datasets[nuscenes]'  # nuscenes-devkit も入れる（Adapter を使う場合）
```

`jidohub.datasets` を import すると、Pillow ベースの既定画像デコーダが登録されます
（既に別のデコーダが登録済みの場合は上書きしません）。

## 使い方

```python
from nuscenes import NuScenes
from jidohub.datasets.nuscenes import NuScenesAdapter

nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes")
adapter = NuScenesAdapter(nusc)

sample = adapter.get_sample(adapter.sample_ids[0])  # 標準スキーマの Sample（入力のみ）
gt = adapter.get_ground_truth(sample.sample_id)  # ego 座標系の Detection3DOutput（GT）
```

Adapter は devkit / CAN bus を内部で生成せず**注入**します（テストで差し替えられるように）。

```python
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

adapter = NuScenesAdapter(
    nusc,
    can_bus=NuScenesCanBus(dataroot="/data/nuscenes"),  # ego_state を埋める（別ダウンロード）
    history_length=2,  # Sample.history に過去 keyframe を 2 件
    command_horizon_s=3.0,  # Sample.command を将来 3 秒から推定（後述）
)
```

### 画像の表現とデコーダの注入

`get_sample` が返す画像は既定で **`encoded`**（JPEG バイト列のまま）です。生画素へのデコードは
`frame.image` にアクセスした時点で初めて走り、以後キャッシュされます。プロセス境界を越える経路では
`encoded` の方が約 1/15 のサイズで運べます。構築時に生画素が必要な場合は
`NuScenesAdapter(..., image_mode="pixels")` を指定します。nvJPEG など高速なデコーダを使いたい場合は、
`jidohub.datasets` を import する前に `jidohub.core.schemas.register_image_decoder(...)` で登録すれば
既定の Pillow デコーダは登録されません（上書きしない方針）。

### `command`（走行指令）について

`Sample.command` は `command_horizon_s` を指定したときのみ埋まります。これは**正解ラベルではなく、
将来の自車位置から推定した値**です（nuScenes に走行指令の GT は存在しません。UniAD 系の慣例に倣い、
将来の横方向変位から直進 / 左折 / 右折を推定します）。実車では航法系から与えられるものであり、
評価に用いる場合はこの定義の違いが結果に影響し得ます。

## fixture の生成（実データが必要・コミット禁止）

jidohub-agents の一致テスト用に、実データから `Sample` / `Detection3DOutput` を直列化した
fixture を生成できます。

```bash
python scripts/make_fixtures.py \
    --dataroot /data/nuscenes --version v1.0-mini \
    --out fixtures --num-samples 5
```

**生成物はリポジトリにコミットしないでください。** nuScenes のデータは再配布に制約のある
ライセンス（非商用）であり、生成した fixture には実画像のバイト列が含まれます。出力先の既定
`fixtures/` は `.gitignore` 済みです。fixture は各自の環境でこのスクリプトから生成してください
（CI ではテスト用の stub で完結しており、fixture は使いません）。

## 開発

```bash
pip install -e '.[dev,nuscenes]'
pytest                 # 変換ロジックと Adapter のユニットテスト
ruff check . && ruff format --check . && mypy
```

`tests/test_conversions.py` は **devkit なし**で通ります（変換ロジックが devkit 非依存である
ことの検証）。`tests/test_adapter.py` も偽 devkit の stub で完結し、nuScenes 本体を必要としません。
設計上の判断（変換ロジックの純粋関数への隔離、nuScenes と標準スキーマの差異）は `CLAUDE.md` を参照。
