"""データセット Adapter の抽象基底。

Adapter の責務は「データセット固有の形式 → 標準スキーマ」の変換に限る。
モデル固有の前処理、可視化、評価は行わない。

設計方針
    - **添字アクセスと反復**を提供する（``len`` / ``__getitem__`` / ``__iter__``）。
      評価ハーネスやベンチマークが素直に書けるようにするため。
    - **GT の取得を分離する**。``get_sample`` は入力のみを返し、
      正解アノテーションは ``get_ground_truth`` で別途取得する。
      推論時に GT を誤って渡す事故を防ぐため、同じオブジェクトに同梱しない。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

from jidohub.core.schemas import Detection3DOutput, Sample

__all__ = ["DatasetAdapter", "ImageMode"]

ImageMode = Literal["encoded", "pixels"]
"""``Sample`` に載せる画像の表現。

- ``"encoded"``（既定）: JPEG バイト列のまま。デコードは利用側が ``frame.image.array`` に触れた時点で走る
- ``"pixels"``: 構築時にデコードして生画素を載せる
"""


class DatasetAdapter(ABC):
    """データセットを標準スキーマに正規化する Adapter の基底クラス。"""

    @property
    @abstractmethod
    def sample_ids(self) -> list[str]:
        """全サンプルの識別子。データセット内の順序を保つこと。"""

    @abstractmethod
    def get_sample(self, sample_id: str) -> Sample:
        """識別子から :class:`Sample` を構築する。GT は含めない。"""

    @abstractmethod
    def get_ground_truth(self, sample_id: str) -> Detection3DOutput:
        """識別子に対応する正解アノテーションを返す。

        座標系は :class:`Sample` と同じ ego 座標系に揃えること。
        """

    @abstractmethod
    def scene_ids(self) -> list[str]:
        """全シーンの識別子。"""

    @abstractmethod
    def samples_in_scene(self, scene_id: str) -> list[str]:
        """シーン内のサンプル識別子を**時刻の昇順**で返す。"""

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Sample:
        return self.get_sample(self.sample_ids[index])

    def __iter__(self) -> Iterator[Sample]:
        for sample_id in self.sample_ids:
            yield self.get_sample(sample_id)
