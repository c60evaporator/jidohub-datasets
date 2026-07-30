"""nuScenes Adapter。

devkit のレコードから値を取り出し、:mod:`jidohub.datasets.nuscenes.conversions` の
純粋関数に渡して標準スキーマを組み立てる。

**このモジュールに計算式を書かないこと**（CLAUDE.md 2.1）。
座標変換・寸法の並べ替え・速度の変換はすべて conversions 側の責務であり、
ここに書くとテストが devkit と実データを要求するようになる。
numpy は :func:`read_lidar_points` の生ファイル読み込み（``np.fromfile``）にのみ使う。
整形・座標計算は行わず、すべて conversions の関数を経由する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jidohub.core.schemas import (
    CameraFrame,
    CoordinateFrame,
    Detection3DOutput,
    DrivingCommand,
    EgoState,
    EncodedImage,
    ImageFormat,
    LidarSweep,
    Sample,
    decode_image,
)

from jidohub.datasets.base import DatasetAdapter, ImageMode
from jidohub.datasets.nuscenes.conversions import (
    LIDAR_FIELDS,
    box_from_annotation,
    driving_command_from_future_positions,
    ego_state_from_can,
    future_ego_positions,
    intrinsic_from_devkit,
    nearest_can_message,
    points_from_pcd_bin,
    transform_from_pose,
)

__all__ = ["NuScenesAdapter", "read_image_bytes", "read_lidar_points"]

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
        can_bus: ``nuscenes.can_bus.can_bus_api.NuScenesCanBus`` 相当のオブジェクト。
            CAN bus は別ダウンロードで利用できない環境が普通にあるため、既定 ``None``。
            ``None`` の場合や該当シーンのデータが無い場合、``ego_state`` は ``None`` になる。
            **nusc と同様 Adapter 内では生成しない**（注入する）。
        image_mode: 画像の表現。既定の ``"encoded"`` は JPEG バイト列のまま載せる。
        history_length: ``Sample.history`` に含める過去 keyframe 数。
            sweep（非キーフレーム）は扱わない（CLAUDE.md 4 章）。
        cameras: 読み込むカメラチャンネル。センサ構成を絞る場合に指定する。
        command_horizon_s: ``Sample.command``（高レベル走行指令）を導出する将来ホライズン[s]。
            ``None``（既定）なら導出せず ``command`` は ``None`` のまま。秒数を指定した場合のみ、
            その範囲の将来 keyframe から**推定**する。``command`` は正解ラベルではなく
            将来の自車位置からの推定値である点に注意（実車では航法系から与えられる）。

    Note:
        追跡 ID の写像スコープは **Adapter インスタンス全体**。``instance_token``
        （nuScenes 全体で一意）を初出順に整数へ写像するため、同一シーン内で一貫し、
        かつシーンをまたいで同じ整数が別の物体を指すことはない。

    Example:
        >>> from nuscenes import NuScenes
        >>> nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes")
        >>> adapter = NuScenesAdapter(nusc)
        >>> sample = adapter.get_sample(adapter.sample_ids[0])
        >>> gt = adapter.get_ground_truth(sample.sample_id)
    """

    def __init__(
        self,
        nusc: Any,
        *,
        can_bus: Any = None,
        image_mode: ImageMode = "encoded",
        history_length: int = 0,
        cameras: tuple[str, ...] = CAMERA_CHANNELS,
        command_horizon_s: float | None = None,
    ) -> None:
        self.nusc = nusc
        self.can_bus = can_bus
        self.image_mode = image_mode
        self.history_length = history_length
        self.cameras = cameras
        self.command_horizon_s = command_horizon_s
        # instance_token -> int の写像。Adapter 全体で一貫（docstring 参照）。
        self._track_ids: dict[str, int] = {}

    # -----------------------------------------------------------------
    # DatasetAdapter の実装
    # -----------------------------------------------------------------

    @property
    def sample_ids(self) -> list[str]:
        return [record["token"] for record in self.nusc.sample]

    def scene_ids(self) -> list[str]:
        return [record["token"] for record in self.nusc.scene]

    def samples_in_scene(self, scene_id: str) -> list[str]:
        scene = self.nusc.get("scene", scene_id)
        tokens: list[str] = []
        token = scene["first_sample_token"]
        # first_sample_token から next を辿る。timestamp で再ソートしない
        # （連結リストの順序が正。CLAUDE.md 3 章 / 依頼書 3.1）。
        while token:
            tokens.append(token)
            token = self.nusc.get("sample", token)["next"]
        return tokens

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
        return self._build_sample(sample_id, history_length=self.history_length)

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
        sample = self.nusc.get("sample", sample_id)
        ego_to_global = self._sample_ego_to_global(sample)
        boxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            # box_velocity は global 座標系。前後の keyframe が無いと NaN を含む。
            # NaN の None 化と座標変換は box_from_annotation の責務（前処理しない）。
            velocity_global = self.nusc.box_velocity(ann_token)
            box = box_from_annotation(
                translation_global=ann["translation"],
                size_wlh=ann["size"],
                rotation_global=ann["rotation"],
                label=ann["category_name"],
                ego_to_global=ego_to_global,
                velocity_global=velocity_global,
                track_id=self._track_id(ann["instance_token"]),
                attributes=self._annotation_attributes(ann),
            )
            boxes.append(box)
        return Detection3DOutput(boxes=boxes, frame=CoordinateFrame.EGO)

    # -----------------------------------------------------------------
    # 内部ヘルパ
    # -----------------------------------------------------------------

    def _build_sample(self, sample_token: str, *, history_length: int) -> Sample:
        """1 つの ``sample`` から :class:`Sample` を組み立てる（履歴段数を制御）。"""
        sample = self.nusc.get("sample", sample_token)
        data = sample["data"]
        ego_to_global = self._sample_ego_to_global(sample)

        cameras = {}
        for channel in self.cameras:
            sd_token = data.get(channel)
            if sd_token is None:
                continue
            cameras[channel] = self._build_camera_frame(sd_token)

        lidar = self._build_lidar_sweep(data[LIDAR_CHANNEL]) if LIDAR_CHANNEL in data else None
        ego_state = self._build_ego_state(sample)
        command = self._driving_command(sample, ego_to_global)

        history: list[Sample] = []
        if history_length > 0:
            # prev を history_length 個まで遡る。遡った Sample は履歴を持たない（1 段のみ）。
            collected: list[str] = []
            prev_token = sample["prev"]
            while prev_token and len(collected) < history_length:
                collected.append(prev_token)
                prev_token = self.nusc.get("sample", prev_token)["prev"]
            # collected は [直近の過去, ..., 古い]。history は時刻昇順（末尾が直近）。
            for token in reversed(collected):
                history.append(self._build_sample(token, history_length=0))

        return Sample(
            timestamp=sample["timestamp"],
            ego_to_global=ego_to_global,
            cameras=cameras,
            lidar=lidar,
            ego_state=ego_state,
            command=command,
            history=history,
            sample_id=sample_token,
            scene_id=sample["scene_token"],
        )

    def _sample_ego_to_global(self, sample: dict) -> Any:
        """``sample`` の LIDAR_TOP ego_pose から ``ego_to_global`` を作る。"""
        lidar_sd = self.nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        return transform_from_pose(ego_pose["translation"], ego_pose["rotation"])

    def _build_camera_frame(self, sample_data_token: str) -> CameraFrame:
        """カメラの ``sample_data`` から :class:`CameraFrame` を作る。

        - ``calibrated_sensor`` から :func:`transform_from_pose` で ``sensor_to_ego``
        - ``camera_intrinsic`` を :func:`intrinsic_from_devkit` で ``intrinsic`` に入れる
        - **画像サイズは ``sample_data`` の ``width`` / ``height`` を使う。**
          サイズを得るために画像を開かないこと（CLAUDE.md 2.3）
        - ``image_mode == "encoded"`` ならファイルのバイト列をそのまま
          :meth:`EncodedImage.from_bytes` に渡す。デコードしない
        """
        sd = self.nusc.get("sample_data", sample_data_token)
        calib = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sensor_to_ego = transform_from_pose(calib["translation"], calib["rotation"])
        intrinsic = intrinsic_from_devkit(calib["camera_intrinsic"])

        data, image_format = read_image_bytes(self.nusc.get_sample_data_path(sample_data_token))
        # サイズは sample_data の値を使う（画像を開かない。CLAUDE.md 2.3）。
        encoded = EncodedImage.from_bytes(
            data, image_format, height=sd["height"], width=sd["width"]
        )
        common = dict(
            intrinsic=intrinsic,
            sensor_to_ego=sensor_to_ego,
            channel=sd["channel"],
            timestamp=sd["timestamp"],
        )
        if self.image_mode == "pixels":
            # 明示的に生画素が要求された場合のみデコードする。
            return CameraFrame(pixels=decode_image(encoded), **common)
        return CameraFrame(encoded=encoded, **common)

    def _build_lidar_sweep(self, sample_data_token: str) -> LidarSweep:
        """LiDAR の ``sample_data`` から :class:`LidarSweep` を作る。

        - :func:`read_lidar_points` が生ファイルを ``np.fromfile`` で読み、
          :func:`points_from_pcd_bin` で ``(N, 5)`` に整形する
          （devkit の ``LidarPointCloud`` は ring_index を落とすため使わない。CLAUDE.md 3 章）
        - ``fields`` は :data:`LIDAR_FIELDS` を使う
        - **点群はセンサ座標系のまま**保持する（ego へ変換しない）
        """
        sd = self.nusc.get("sample_data", sample_data_token)
        calib = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sensor_to_ego = transform_from_pose(calib["translation"], calib["rotation"])
        raw = read_lidar_points(self.nusc.get_sample_data_path(sample_data_token))
        return LidarSweep(
            points=points_from_pcd_bin(raw),
            sensor_to_ego=sensor_to_ego,
            channel=sd["channel"],
            fields=LIDAR_FIELDS,
            timestamp=sd["timestamp"],
        )

    def _build_ego_state(self, sample: dict) -> EgoState | None:
        """CAN bus 情報から :class:`EgoState` を作る。

        nuScenes の CAN bus 拡張は別ダウンロードで、利用できない環境が普通にある。
        ``can_bus`` が ``None`` の場合、該当シーンのデータが無い場合、または最寄りの
        メッセージが時刻的に離れすぎている場合は ``None`` を返す
        （速度を 0 で埋めない。「静止している」という誤情報になるため。
        また ego_pose 差分からの推定もしない。推定値を実測と区別なく返さない）。

        取得できる場合の対応（詳細は :func:`ego_state_from_can`）
            - 速度・加速度・角速度は ``pose`` メッセージ（ego 座標系・SI 単位）から取る
            - ``steering_angle`` は ``vehicle_monitor`` の ``steering``（度）を rad へ変換
        """
        if self.can_bus is None:
            return None
        scene = self.nusc.get("scene", sample["scene_token"])
        scene_name = scene["name"]
        timestamp = sample["timestamp"]
        try:
            pose_messages = self.can_bus.get_messages(scene_name, "pose")
            monitor_messages = self.can_bus.get_messages(scene_name, "vehicle_monitor")
        except Exception:
            # 該当シーンの CAN bus データが無い場合、devkit は例外を送出する。
            return None
        pose_msg = nearest_can_message(pose_messages, timestamp)
        if pose_msg is None:
            return None
        monitor_msg = nearest_can_message(monitor_messages, timestamp)
        steering_deg = None if monitor_msg is None else monitor_msg["steering"]
        return ego_state_from_can(pose_msg, steering_deg)

    def _driving_command(self, sample: dict, ego_to_global: Any) -> DrivingCommand | None:
        """``command_horizon_s`` 指定時のみ、将来 keyframe から走行指令を推定する。"""
        if self.command_horizon_s is None:
            return None
        horizon_us = self.command_horizon_s * 1_000_000
        base_timestamp = sample["timestamp"]
        future_ego_to_global = []
        token = sample["next"]
        while token:
            record = self.nusc.get("sample", token)
            if record["timestamp"] - base_timestamp > horizon_us:
                break
            future_ego_to_global.append(self._sample_ego_to_global(record))
            token = record["next"]
        if not future_ego_to_global:
            # 将来フレームが 1 つも無ければ None（終端で不足しても失敗させない）。
            return None
        positions = future_ego_positions(ego_to_global, future_ego_to_global)
        return driving_command_from_future_positions(positions)

    def _track_id(self, instance_token: str) -> int:
        """``instance_token`` を Adapter 全体で一貫した整数へ写像する。"""
        track_id = self._track_ids.get(instance_token)
        if track_id is None:
            track_id = len(self._track_ids)
            self._track_ids[instance_token] = track_id
        return track_id

    def _annotation_attributes(self, annotation: dict) -> dict:
        """``sample_annotation`` の属性トークンを名前へ解決して dict に載せる。"""
        names = [
            self.nusc.get("attribute", token)["name"]
            for token in annotation.get("attribute_tokens", [])
        ]
        return {"nuscenes_attributes": names}


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


def read_lidar_points(path: str | Path) -> Any:
    """LiDAR の ``.pcd.bin`` を読み、フラットな ``float32`` 配列を返す。

    整形（``(N, 5)`` 化）は
    :func:`~jidohub.datasets.nuscenes.conversions.points_from_pcd_bin` の責務。
    **devkit の ``LidarPointCloud`` は使わない**（ring_index を切り捨てて 4 列にするため。
    CLAUDE.md 3 章）。パスの解決は ``nusc.get_sample_data_path`` で行い、ここでは
    読み込みだけを行う。この関数を差し替えることで、テストは実ファイル無しで動かせる。
    """
    return np.fromfile(str(path), dtype=np.float32)
