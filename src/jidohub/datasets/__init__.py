"""jidohub-datasets: データセットを標準スキーマに正規化する Python API。

core にのみ依存し、agents / interfaces には依存しない（星形依存）。

import 時に Pillow ベースの既定画像デコーダを登録する。
**既に登録済みの場合は上書きしない**（利用者が nvJPEG 等を先に登録している
可能性があるため。CLAUDE.md 2.4）。
"""

from __future__ import annotations

from jidohub.datasets.base import DatasetAdapter, ImageMode
from jidohub.datasets.decoders import pillow_decoder, register_default_decoder

register_default_decoder()

__all__ = [
    "DatasetAdapter",
    "ImageMode",
    "pillow_decoder",
    "register_default_decoder",
]
