"""画像デコーダの実装と登録ポリシー。

core は画像コーデックに依存しないため、``EncodedPixels`` のデコードは
利用側が :func:`jidohub.core.schemas.register_image_decoder` で注入する。
本リポジトリは Pillow ベースの既定デコーダを提供する。

登録ポリシー
    ``jidohub.datasets`` の import 時に自動登録するが、**既に登録済みなら上書きしない**。
    利用者が nvJPEG など高速なデコーダを先に登録している可能性があるため
    （CLAUDE.md 2.4）。明示的に差し替えたい場合は
    :func:`jidohub.core.schemas.register_image_decoder` を直接呼ぶ。
"""

from __future__ import annotations

import importlib.util
import io

import numpy as np
from jidohub.core.schemas import EncodedPixels, get_image_decoder, register_image_decoder

__all__ = ["pillow_decoder", "register_default_decoder"]


def pillow_decoder(encoded: EncodedPixels) -> np.ndarray:
    """Pillow で符号化画像をデコードする。

    :data:`~jidohub.core.schemas.ImageDecoder` の契約通り、
    shape ``(H, W, 3)`` の ``np.uint8`` を **RGB 順**で返す。
    グレースケールや RGBA の画像も RGB に変換する
    （契約が常に 3 チャンネル RGB であるため）。
    """
    from PIL import Image

    with Image.open(io.BytesIO(encoded.to_bytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def register_default_decoder(force: bool = False) -> bool:
    """既定デコーダ（Pillow）を登録する。

    Pillow が import できる場合のみ登録する。Pillow の無い環境で登録してしまうと、
    ``frame.image`` アクセス時に ``pillow_decoder`` 内部で ``ImportError`` が送出され、
    core の :class:`~jidohub.core.schemas.ImageDecodeError`（「デコーダが未登録」）に
    ならないため、利用者に原因が伝わらない（依頼書 タスク 1.2）。
    Pillow が無い場合は登録せず ``False`` を返し、``frame.image`` は
    「デコーダ未登録」の ``ImageDecodeError`` になる。

    Args:
        force: ``True`` なら既存の登録を上書きする。

    Returns:
        登録した場合は ``True``、既存の登録を尊重してスキップした場合、
        または Pillow が無く登録できなかった場合は ``False``。
    """
    if importlib.util.find_spec("PIL") is None:
        return False
    if not force and get_image_decoder() is not None:
        return False
    register_image_decoder(pillow_decoder)
    return True
