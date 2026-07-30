# CLAUDE.md — jidohub-datasets

このファイルには「ソースやconfigを読んでも分からない設計判断」だけを記載する。
依存バージョン・ディレクトリの中身・コマンドの詳細は `pyproject.toml` / 各モジュールを参照すること。

---

## 1. このリポジトリの位置づけ

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォーム。
**jidohub-datasets は「データセット固有の形式 → 標準スキーマ」の変換を担う**。

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストするWebプラットフォーム |
| jidohub-core | 標準スキーマ・Hubクライアント・configパーサ |
| jidohub-agents | Agentをロードして実行するPython API |
| **jidohub-datasets（本リポジトリ）** | Datasetをロードして `Sample` に正規化する |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

### 依存の原則

- **core にのみ依存する。** agents / interfaces / web に依存してはならない。
- core は datasets に依存しない（星形依存の一方向を守る）。
- **`torch` に依存しない。** データセットの読み込みに学習フレームワークは不要であり、
  torch なしで可視化・変換ができることがプラットフォーム全体の前提。

### ディレクトリ構成と各ファイルの責務

```
src/
└── jidohub/                     ← namespace package。__init__.py を置かない
    └── datasets/
        ├── __init__.py          再エクスポート + 既定デコーダの登録
        ├── base.py              DatasetAdapter の抽象基底
        ├── decoders.py          画像デコーダ（Pillow）と登録ポリシー
        └── nuscenes/
            ├── __init__.py
            ├── conversions.py   **devkit に依存しない純粋関数**（変換ロジックの中核）
            └── adapter.py       devkit からの読み出しと conversions の呼び出し
```

配置ルール

- **`conversions.py` に devkit を import しない。** 座標系・寸法順・速度の変換は
  バグの温床であり、ここを純粋関数に保つことで **devkit も実データもなしに
  ユニットテストできる**（2 章参照）。この分離が本リポジトリの設計の要。
- **`adapter.py` は変換ロジックを持たない。** devkit のレコードから値を取り出し、
  `conversions.py` の関数に渡すだけにする。計算式を adapter に書かない。
- 新しいデータセットを追加する場合は `nuscenes/` と同じ構造
  （`conversions.py` + `adapter.py`）を作る。

---

## 2. 絶対に守る規約

### 2.1 変換ロジックは純粋関数に隔離する（最重要）

nuScenes と標準スキーマの間には**取り違えても例外が出ない差異**が複数ある（3 章）。
これらを検出できる唯一の方法は、既知の入力に対する既知の出力を検証するテストである。

- 変換は `conversions.py` の純粋関数として書く。副作用・devkit 依存を持たせない
- **各変換関数には、手計算で検証できるテストを必ず書く**
  （例: 単位回転・既知の平行移動での global → ego 変換）
- テストは devkit と実データなしで通ること。CI にデータセットを置かない

### 2.2 標準スキーマを再定義しない

`Sample` / `Box3D` などの型は core が唯一の正。
datasets 側で同等の型を定義したり、フィールドを追加したりしない。
表現できない情報が出てきた場合は、`metadata` に逃がす前に
**core のスキーマを拡張すべきかを報告すること**。

### 2.3 画像はデコードせずに運ぶ（既定）

`Sample` を構築する際、画像は既定で `EncodedImage`（JPEG バイト列のまま）にする。

- nuScenes の画像はディスク上で既に JPEG。読んだバイト列をそのまま載せる
- デコードは利用側が `frame.image` にアクセスした時点で初めて走る
- 生画素が必要な場合のみ `image_mode="pixels"` を指定できるようにする
- **画像サイズは `sample_data` レコードの `width` / `height` を使う。**
  サイズを得るために画像を開かない

### 2.4 デコーダの登録は上書きしない

`jidohub.datasets` の import 時に Pillow ベースのデコーダを登録するが、
**既に登録済みの場合は上書きしない**。利用者が nvJPEG など高速なデコーダを
登録している可能性があるため。登録ポリシーは `decoders.py` に集約する。

### 2.5 点群はセンサ座標系のまま

core の規約通り、`LidarSweep.points` はセンサ座標系で保持する。
ego 座標への変換は利用側が `points_in_ego()` で行う。
Adapter で ego に変換して渡さないこと。

---

## 3. nuScenes と標準スキーマの差異（実装ミスが最も起きやすい箇所）

**変換を書く前に必ずこの表を確認すること。** いずれも取り違えても例外が出ない。

| 項目 | nuScenes | 標準スキーマ | 対応 |
|---|---|---|---|
| ボックス寸法 | `size = [width, length, height]` | `size = (length, width, height)` | **入れ替える**。`Box3D.from_dimensions()` を使えば取り違えない |
| ボックス座標系 | global | ego（既定） | `inv(ego_to_global)` を適用 |
| 速度 | global 座標系、値が `NaN` になることがある | `frame` と同じ座標系 | 回転成分のみ適用。`NaN` は `None` に落とす |
| 点群の形状 | 生 `.pcd.bin` は点ごとに 5 値が連続（行優先） | `(N, C)` | 生ファイルを読み `(-1, 5)` に整形（**転置は不要**）。`points_from_pcd_bin` を使う |
| 点群の列 | x, y, z, intensity, ring_index（生ファイルは 5 列） | 先頭 3 列が x, y, z | `fields` に名前を宣言する。**devkit の `LidarPointCloud.from_file` は `ring_index` を切り捨てて 4 列にするため使わない**（`nbr_dims()==4`）。生 `.pcd.bin` を直接読み 5 列を保持する |
| 時刻 | マイクロ秒 int | マイクロ秒 int | 変換不要 |
| 回転表現 | quaternion `(w, x, y, z)` | 同じ | 変換不要 |
| ボックス中心 | 幾何中心 | 幾何中心 | 変換不要 |
| 変換行列 | `translation` + `rotation` の組 | 4x4 同次変換行列 | 組み立てる。**向きに注意** |

変換行列の向き（間違えやすい）

- `ego_pose` → **ego → global**（`Sample.ego_to_global` にそのまま入る）
- `calibrated_sensor` → **sensor → ego**（`CameraFrame.sensor_to_ego` にそのまま入る）
- どちらも「その座標系の点を親座標系へ移す」向き。逆向きが必要な場合は
  `jidohub.core.geometry.invert_transform` を使い、自前で転置を書かない

---

## 4. スコープ（当面やらないこと）

以下は**実装せず、必要になった時点で報告して指示を仰ぐこと**。

- **sweep（非キーフレーム）の対応**。当面は keyframe（2Hz の sample）のみ扱う。
  `Sample.history` も keyframe を遡る
- **複数スイープの点群集約**。モデル側の前処理の責務
- **lidarseg / panoptic などの追加アノテーション**
- **PostGIS（nuscenes-viewer の DB）からの Adapter**。devkit 版を先に完成させる
  （viewer 非依存でテストでき、正解の基準になる）
- **学習フレームワーク形式への Exporter**（mmdet3d の info file 等）
- **データセットのアップロード / バージョニング**（jidohub-web 稼働後）

---

## 5. 行動原則

- 3 章の差異に関わるコードを書くときは、**推測で実装しない**。
  表にない差異を見つけた場合は、実装する前に表に追記して報告する
- 変換関数を追加したら、**手計算で検証できるテストを同時に書く**。
  「実データで動いたから正しい」は根拠にならない（取り違えても動くため）
- devkit の API が期待と違う場合、**Adapter 側で辻褄を合わせず報告する**。
  変換の意味を変える対処は設計判断であり、実装で埋めるべきではない
- 依存ライブラリの追加、`Sample` に載せられない情報の発見、
  スコープ外（4 章）への進出は自己判断で行わない
