"""実データから ``Sample`` / ``Detection3DOutput`` を直列化した fixture を生成する。

jidohub-agents の一致テスト（Adapter が生成する標準スキーマとモデル入出力の突き合わせ）で
使うことを想定している。**生成物はリポジトリにコミットしないこと。** nuScenes のデータは
再配布に制約のあるライセンス（非商用）であり、生成した fixture には実画像のバイト列が含まれる。
出力先の既定 ``fixtures/`` は ``.gitignore`` 済み。CI ではこのスクリプトを使わず、
テストは偽 devkit の stub で完結する。

使い方::

    python scripts/make_fixtures.py \\
        --dataroot /data/nuscenes --version v1.0-mini \\
        --out fixtures --num-samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jidohub.core.serialization import pack

from jidohub.datasets.nuscenes.adapter import NuScenesAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, help="nuScenes の dataroot")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes のバージョン")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures"),
        help="出力先ディレクトリ（既定 fixtures/ は .gitignore 済み）",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="生成するサンプル数")
    args = parser.parse_args()

    # devkit はここでだけ必要（jidohub-datasets[nuscenes] を入れること）。
    from nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    adapter = NuScenesAdapter(nusc)

    args.out.mkdir(parents=True, exist_ok=True)
    for sample_id in adapter.sample_ids[: args.num_samples]:
        sample = adapter.get_sample(sample_id)
        ground_truth = adapter.get_ground_truth(sample_id)
        (args.out / f"{sample_id}.sample.bin").write_bytes(pack(sample))
        (args.out / f"{sample_id}.gt.bin").write_bytes(pack(ground_truth))
        print(f"wrote {sample_id}")

    print(f"done: {args.num_samples} サンプルを {args.out} に生成した（コミットしないこと）")


if __name__ == "__main__":
    main()
