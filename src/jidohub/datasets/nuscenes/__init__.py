"""nuScenes 対応。

``adapter.py`` は devkit を **import しない**（``nusc`` を注入する設計）。
そのため ``NuScenesAdapter`` は devkit 未インストールの環境でも import できる。
devkit が必要になるのは、利用者が ``NuScenes`` インスタンスを生成する時点だけ。
"""

from __future__ import annotations

from jidohub.datasets.nuscenes.adapter import (
    CAMERA_CHANNELS,
    LIDAR_CHANNEL,
    NuScenesAdapter,
    read_image_bytes,
    read_lidar_points,
)
from jidohub.datasets.nuscenes.conversions import (
    LIDAR_FIELDS,
    NUSCENES_LIDAR_COLUMNS,
    box_from_annotation,
    driving_command_from_future_positions,
    ego_state_from_can,
    future_ego_positions,
    intrinsic_from_devkit,
    nearest_can_message,
    points_from_devkit,
    points_from_pcd_bin,
    transform_from_pose,
)

__all__ = [
    "NuScenesAdapter",
    "CAMERA_CHANNELS",
    "LIDAR_CHANNEL",
    "read_image_bytes",
    "read_lidar_points",
    "LIDAR_FIELDS",
    "NUSCENES_LIDAR_COLUMNS",
    "transform_from_pose",
    "box_from_annotation",
    "points_from_devkit",
    "points_from_pcd_bin",
    "driving_command_from_future_positions",
    "intrinsic_from_devkit",
    "nearest_can_message",
    "ego_state_from_can",
    "future_ego_positions",
]
