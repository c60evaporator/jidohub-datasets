"""nuScenes 対応。

devkit（``nuscenes-devkit``）は optional 依存であり、
:mod:`jidohub.datasets.nuscenes.adapter` の import 時にのみ必要になる。
変換ロジック（:mod:`jidohub.datasets.nuscenes.conversions`）は devkit なしで使える。
"""

from __future__ import annotations

from jidohub.datasets.nuscenes.conversions import (
    LIDAR_FIELDS,
    box_from_annotation,
    driving_command_from_future_positions,
    points_from_devkit,
    transform_from_pose,
)

__all__ = [
    "LIDAR_FIELDS",
    "transform_from_pose",
    "box_from_annotation",
    "points_from_devkit",
    "driving_command_from_future_positions",
]
