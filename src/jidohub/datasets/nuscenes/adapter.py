"""nuScenes Adapter。

devkit のレコードから値を取り出し、:mod:`jidohub.datasets.nuscenes.conversions` の
純粋関数に渡して標準スキーマを組み立てる。

**このモジュールに計算式を書かないこと**（CLAUDE.md 2.1）。
座標変換・寸法の並べ替え・速度の変換はすべて conversions 側の責務であり、
ここに書くとテストが devkit と実データを要求するようになる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jidohub.core.schemas import (
    CameraFrame,
    Detection3DOutput,
    EgoState,
    EncodedImage,
    ImageFormat,
    LidarSweep,
    Sample,
)
from jidohub.datasets.base import DatasetAdapter, ImageMode
from jidohub.datasets.nuscenes.conversions import (
    LIDAR_FIELDS,
    box_from_annotation,
    points_from_devkit,
    transform_from_pose,
)

__all__ = ["NuScenesAdapter"]

LIDAR_CHANNEL = "LIDAR_TOP"
CAMERA_CHANNELS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


class NuScenesAdapter(DatasetAdapter):
    """nuscenes-devkit を用いた Adapter。

    Args:
        nusc: ``nuscenes.NuScenes`` のインスタンス。**Adapter は devkit を生成しない**
            （バージョン・パス・verbose の指定は呼び出し側の関心事であり、
            テストで差し替えられるようにするため）。
        image_mode: 画像の表現。既定の ``"encoded"`` は JPEG バイト列のまま載せる。
        history_length: ``Sample.history`` に含める過去 keyframe 数。
            sweep（非キーフレーム）は扱わない（CLAUDE.md 4 章）。
        cameras: 読み込むカメラチャンネル。センサ構成を絞る場合に指定する。

    Example:
        >>> from nuscenes import NuScenes
        >>> nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes")
        >>> adapter = NuScenesAdapter(nusc)
        >>> sample = adapter.get_sample(adapter.sample_ids[0])
        >>> gt = adapter.get_ground_truth(sample.sample_id)
    """

    def __init__(
        self,
        nusc,
        image_mode: ImageMode = "encoded",
        history_length: int = 0,
        cameras: tuple[str, ...] = CAMERA_CHANNELS,
    ) -> None:
        self.nusc = nusc
        self.image_mode = image_mode
        self.history_length = history_length
        self.cameras = cameras

    # -----------------------------------------------------------------
    # DatasetAdapter の実装
    # -----------------------------------------------------------------

    @property
    def sample_ids(self) -> list[str]:
        raise NotImplementedError(
            "nusc.sample を走査し、sample_token のリストを返す。"
            "順序は devkit の格納順（シーン単位で時刻昇順）を保つこと。"
        )

    def scene_ids(self) -> list[str]:
        raise NotImplementedError("nusc.scene から scene_token のリストを返す。")

    def samples_in_scene(self, scene_id: str) -> list[str]:
        raise NotImplementedError(
            "scene の first_sample_token から next を辿り、時刻昇順のリストを返す。"
            "sample レコードの timestamp でソートし直さないこと（連結リストの順序が正）。"
        )

    def get_sample(self, sample_id: str) -> Sample:
        """``sample_token`` から :class:`Sample` を構築する。

        手順
            1. ``sample`` レコードを取得
            2. LIDAR_TOP の ``sample_data`` から ``ego_pose`` を引き、
               :func:`transform_from_pose` で ``ego_to_global`` を作る
            3. 各カメラの ``sample_data`` から :class:`CameraFrame` を作る
            4. LiDAR の ``sample_data`` から :class:`LidarSweep` を作る
            5. ``history_length`` 分だけ ``prev`` を遡って ``history`` を埋める
               （遡った Sample はさらに history を持たせない。1 段のみ）

        Note:
            ``ego_to_global`` の基準は **LIDAR_TOP の ego_pose** とする。
            nuScenes ではセンサごとに取得時刻が異なり ego_pose も異なるため、
            どれを Sample 全体の代表とするかを決めておく必要がある。
            各センサ固有の時刻は ``CameraFrame.timestamp`` 等に保持する。
        """
        raise NotImplementedError

    def get_ground_truth(self, sample_id: str) -> Detection3DOutput:
        """``sample_annotation`` を ego 座標系の :class:`Detection3DOutput` に変換する。

        手順
            1. ``sample["anns"]`` の各 ``sample_annotation`` を取得
            2. 速度は ``nusc.box_velocity(annotation_token)`` で取得（global 座標系、
               前後の keyframe が無い場合は ``NaN`` を含む）
            3. :func:`box_from_annotation` に渡す。**寸法の並べ替えと座標変換は
               この関数が行うため、ここで前処理しない**
            4. ``instance_token`` を追跡 ID に使う場合は、シーン内で一貫した
               整数へ写像する（トークン文字列をそのまま入れない）

        Note:
            ``category_name``（``"vehicle.car"`` 等）をそのまま ``label`` に入れる。
            nuScenes detection challenge の 10 クラスへの集約は**評価側の責務**であり、
            Adapter で情報を落とさない。
        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 内部ヘルパ
    # -----------------------------------------------------------------

    def _build_camera_frame(self, sample_data_token: str) -> CameraFrame:
        """カメラの ``sample_data`` から :class:`CameraFrame` を作る。

        - ``calibrated_sensor`` から :func:`transform_from_pose` で ``sensor_to_ego``
        - ``camera_intrinsic`` をそのまま ``intrinsic`` に入れる
        - **画像サイズは ``sample_data`` の ``width`` / ``height`` を使う。**
          サイズを得るために画像を開かないこと（CLAUDE.md 2.3）
        - ``image_mode == "encoded"`` ならファイルのバイト列をそのまま
          :meth:`EncodedImage.from_bytes` に渡す。デコードしない
        """
        raise NotImplementedError

    def _build_lidar_sweep(self, sample_data_token: str) -> LidarSweep:
        """LiDAR の ``sample_data`` から :class:`LidarSweep` を作る。

        - devkit の ``LidarPointCloud.from_file`` は ``(5, N)`` を返すため、
          :func:`points_from_devkit` で ``(N, 5)`` に転置する
        - ``fields`` は :data:`LIDAR_FIELDS` を使う
        - **点群はセンサ座標系のまま**保持する（ego へ変換しない）
        """
        raise NotImplementedError

    def _build_ego_state(self, sample_data_token: str) -> EgoState | None:
        """CAN bus 情報から :class:`EgoState` を作る。

        nuScenes の CAN bus 拡張が利用できない場合は ``None`` を返すこと
        （速度を 0 で埋めない。「静止している」という誤情報になるため）。

        取得できる場合の対応
            - ``vehicle_monitor`` の速度は **km/h** なので m/s へ変換する
            - CAN bus の座標系は ego 座標系（x 前方）
        """
        raise NotImplementedError


def read_image_bytes(path: str | Path) -> tuple[bytes, ImageFormat]:
    """画像ファイルをバイト列として読み、フォーマットを判定する。

    拡張子ではなく**マジックバイト**で判定する。nuScenes は JPEG だが、
    他データセットで PNG が混在した場合に誤ったフォーマットを宣言しないため。
    """
    data = Path(path).read_bytes()
    if data.startswith(b"\xff\xd8\xff"):
        return data, ImageFormat.JPEG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data, ImageFormat.PNG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, ImageFormat.WEBP
    raise ValueError(f"unsupported image format: {path}")


def _unused() -> None:
    """骨子の段階で未使用の import を保持するためのプレースホルダ。

    実装完了後に削除すること。
    """
    _ = (np, EncodedImage, LIDAR_FIELDS, box_from_annotation, LIDAR_CHANNEL)
