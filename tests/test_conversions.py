"""``conversions.py`` の手計算検証テスト。

**devkit と実データ無しで通ること**（CLAUDE.md 2.1）。nuScenes と標準スキーマの差異
（寸法順・座標系・速度・点群形状）は取り違えても例外が出ないため、既知の入出力を
検証するこのテストだけが唯一の防御になる。

基準シナリオ: 自車が global 座標の ``(10, 20)`` に居て、90° 左（+y 方向）を向いている。
このとき「global で +y 方向 = ego で +x（前方）」「global で -x 方向 = ego で +y（左方）」
になる。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from jidohub.core.geometry import (
    quaternion_to_rotation_matrix,
    quaternion_to_yaw,
    rotation_matrix_to_quaternion,
    yaw_to_quaternion,
)
from jidohub.core.schemas import DrivingCommand

from jidohub.datasets.nuscenes.conversions import (
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


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    """``(x, y, yaw)`` の平面姿勢から ego→global 変換を作る（手計算用）。"""
    return transform_from_pose((x, y, 0.0), yaw_to_quaternion(yaw))


# 自車: global (10, 20)、90° 左（+y）向き。
EGO_AT_10_20_FACING_LEFT = _pose(10.0, 20.0, math.pi / 2)


# ---------------------------------------------------------------------------
# transform_from_pose
# ---------------------------------------------------------------------------


def test_transform_from_pose_maps_ego_forward_to_global_plus_y() -> None:
    transform = EGO_AT_10_20_FACING_LEFT
    point_global = transform[:3, :3] @ np.array([1.0, 0.0, 0.0]) + transform[:3, 3]
    # ego の前方 (1,0,0) は +y 向きなので global では (10, 21, 0)
    np.testing.assert_allclose(point_global, [10.0, 21.0, 0.0], atol=1e-9)


def test_transform_from_pose_rejects_bad_length() -> None:
    with pytest.raises((ValueError, IndexError)):
        transform_from_pose((1.0, 2.0), yaw_to_quaternion(0.0))  # translation が (2,)


# ---------------------------------------------------------------------------
# box_from_annotation
# ---------------------------------------------------------------------------


def _car_box(**overrides: object):
    kwargs = dict(
        translation_global=(10.0, 25.0, 0.0),
        size_wlh=(1.9, 4.5, 1.6),  # nuScenes 順 (width, length, height)
        rotation_global=yaw_to_quaternion(math.pi / 2),  # 自車と同じ向き
        label="vehicle.car",
        ego_to_global=EGO_AT_10_20_FACING_LEFT,
    )
    kwargs.update(overrides)
    return box_from_annotation(**kwargs)  # type: ignore[arg-type]


def test_box_center_transformed_to_ego() -> None:
    box = _car_box()
    # global (10,25) は自車の 5m 前方 → ego (5, 0, 0)
    np.testing.assert_allclose(box.center, [5.0, 0.0, 0.0], atol=1e-9)


def test_box_dimensions_reordered_wlh_to_lwh() -> None:
    box = _car_box()
    assert box.length == pytest.approx(4.5)
    assert box.width == pytest.approx(1.9)
    assert box.height == pytest.approx(1.6)
    # 乗用車相当では車長 > 車幅。並べ替えを誤ると成立しない（CLAUDE.md 3 章）。
    assert box.length > box.width


def test_box_orientation_relative_to_ego_is_zero_yaw() -> None:
    box = _car_box()
    assert box.yaw == pytest.approx(0.0, abs=1e-9)


def test_box_velocity_rotated_into_ego_frame() -> None:
    box = _car_box(velocity_global=(0.0, 8.3, 0.0))
    # global +y 8.3 m/s は ego 前方 8.3 m/s → (8.3, 0, 0)
    assert box.velocity is not None
    np.testing.assert_allclose(box.velocity, [8.3, 0.0, 0.0], atol=1e-9)


def test_box_nan_velocity_becomes_none_not_zero() -> None:
    box = _car_box(velocity_global=(float("nan"), float("nan"), float("nan")))
    # 0 で埋めると「静止」という誤情報になるため None（CLAUDE.md 3 章）。
    assert box.velocity is None


def test_box_with_identity_ego_keeps_global_coordinates() -> None:
    identity = np.eye(4)
    box = box_from_annotation(
        translation_global=(7.0, 8.0, 9.0),
        size_wlh=(1.9, 4.5, 1.6),
        rotation_global=yaw_to_quaternion(0.0),
        label="vehicle.car",
        ego_to_global=identity,
        velocity_global=(1.0, 2.0, 3.0),
    )
    np.testing.assert_allclose(box.center, [7.0, 8.0, 9.0], atol=1e-12)
    assert box.velocity is not None
    np.testing.assert_allclose(box.velocity, [1.0, 2.0, 3.0], atol=1e-12)


# ---------------------------------------------------------------------------
# points_from_devkit
# ---------------------------------------------------------------------------


def test_points_from_devkit_transposes_and_casts() -> None:
    # devkit レイアウト (C=5, N=8): 行が x, y, z, intensity, ring
    devkit = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    points = points_from_devkit(devkit)
    assert points.shape == (8, 5)
    assert points.dtype == np.float32
    assert points.flags["C_CONTIGUOUS"]
    # 列の内容が devkit の行と一致すること
    for column in range(5):
        np.testing.assert_allclose(points[:, column], devkit[column, :])


def test_points_from_devkit_rejects_already_transposed() -> None:
    # (N=20, C=5) は shape[0] > shape[1]。転置漏れとして弾く。
    with pytest.raises(ValueError):
        points_from_devkit(np.zeros((20, 5), dtype=np.float32))


def test_points_from_devkit_rejects_1d() -> None:
    with pytest.raises(ValueError):
        points_from_devkit(np.zeros(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# points_from_pcd_bin（生 .pcd.bin。devkit と違い ring_index を保持する）
# ---------------------------------------------------------------------------


def test_points_from_pcd_bin_reshapes_to_5_columns() -> None:
    # 8 点 × 5 列 (x, y, z, intensity, ring_index) のフラット配列
    flat = np.arange(8 * 5, dtype=np.float32)
    points = points_from_pcd_bin(flat)
    assert points.shape == (8, 5)
    assert points.dtype == np.float32
    assert points.flags["C_CONTIGUOUS"]
    # 行優先: 先頭の点は最初の 5 値。ring_index（5 列目）が保持されている。
    np.testing.assert_allclose(points[0], [0, 1, 2, 3, 4])


def test_points_from_pcd_bin_rejects_non_multiple_of_five() -> None:
    with pytest.raises(ValueError):
        points_from_pcd_bin(np.zeros(23, dtype=np.float32))


# ---------------------------------------------------------------------------
# driving_command_from_future_positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final_lateral, expected",
    [
        (0.5, DrivingCommand.GO_STRAIGHT),
        (6.0, DrivingCommand.TURN_LEFT),
        (-6.0, DrivingCommand.TURN_RIGHT),
    ],
)
def test_driving_command_from_lateral_displacement(
    final_lateral: float, expected: DrivingCommand
) -> None:
    future = np.array([[1.0, 0.0], [3.0, final_lateral]])
    assert driving_command_from_future_positions(future) == expected


# ---------------------------------------------------------------------------
# rotation_matrix_to_quaternion（数値安定性 / Shepperd 法の分岐）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw",
    [0.0, math.pi / 4, math.pi / 2, math.pi - 1e-6, -math.pi / 3],
)
def test_quaternion_from_rotation_matrix_roundtrip(yaw: float) -> None:
    # yaw≒π（w が 0 に近い領域）を含め、rotation_matrix_to_quaternion の分岐を通す。
    matrix = quaternion_to_rotation_matrix(yaw_to_quaternion(yaw))
    quaternion = rotation_matrix_to_quaternion(matrix)
    # 符号の自由度があるため成分ではなく回転行列 / yaw で比較する。
    np.testing.assert_allclose(quaternion_to_rotation_matrix(quaternion), matrix, atol=1e-9)
    assert quaternion_to_yaw(quaternion) == pytest.approx(yaw, abs=1e-6)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# intrinsic_from_devkit
# ---------------------------------------------------------------------------


def test_intrinsic_from_devkit_shape_and_dtype() -> None:
    devkit = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
    intrinsic = intrinsic_from_devkit(devkit)
    assert intrinsic.shape == (3, 3)
    assert intrinsic.dtype == np.float64
    np.testing.assert_allclose(intrinsic, devkit)


def test_intrinsic_from_devkit_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        intrinsic_from_devkit([[1.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# nearest_can_message
# ---------------------------------------------------------------------------


def test_nearest_can_message_picks_closest_within_tolerance() -> None:
    messages = [{"utime": 100, "tag": "a"}, {"utime": 200, "tag": "b"}]
    assert nearest_can_message(messages, 130)["tag"] == "a"
    assert nearest_can_message(messages, 170)["tag"] == "b"


def test_nearest_can_message_none_when_too_far() -> None:
    messages = [{"utime": 0}]
    # 既定許容差 0.5 秒（500_000 μs）を超える → None
    assert nearest_can_message(messages, 10_000_000) is None


def test_nearest_can_message_empty_is_none() -> None:
    assert nearest_can_message([], 100) is None


def test_nearest_can_message_custom_tolerance() -> None:
    messages = [{"utime": 0}]
    assert nearest_can_message(messages, 300, max_dt_us=1000) is not None
    assert nearest_can_message(messages, 3000, max_dt_us=1000) is None


# ---------------------------------------------------------------------------
# ego_state_from_can
# ---------------------------------------------------------------------------


def test_ego_state_from_can_maps_pose_fields_and_converts_steering() -> None:
    pose = {"vel": [1.0, 2.0, 3.0], "accel": [0.1, 0.2, 0.3], "rotation_rate": [0.0, 0.0, 0.5]}
    state = ego_state_from_can(pose, steering_deg=90.0)
    np.testing.assert_allclose(state.velocity, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.acceleration, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state.angular_velocity, [0.0, 0.0, 0.5])
    # steering は度 → ラジアン
    assert state.steering_angle == pytest.approx(math.pi / 2)


def test_ego_state_from_can_without_steering() -> None:
    pose = {"vel": [0.0, 0.0, 0.0], "accel": [0.0, 0.0, 0.0], "rotation_rate": [0.0, 0.0, 0.0]}
    state = ego_state_from_can(pose)
    assert state.steering_angle is None


# ---------------------------------------------------------------------------
# future_ego_positions
# ---------------------------------------------------------------------------


def test_future_ego_positions_in_reference_ego_frame() -> None:
    reference = EGO_AT_10_20_FACING_LEFT  # global (10,20), 90° 左向き（非原点・非単位回転）
    future = [
        _pose(10.0, 25.0, math.pi / 2),  # 5m 前方 → ego (5, 0)
        _pose(10.0, 20.0, 0.0),  # 基準位置 → ego (0, 0)（向きは無関係）
        _pose(9.0, 20.0, math.pi),  # global -x 方向 1m → ego 左 (+y) 1m → (0, 1)
    ]
    positions = future_ego_positions(reference, future)
    assert positions.shape == (3, 2)
    np.testing.assert_allclose(positions[0], [5.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(positions[1], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(positions[2], [0.0, 1.0], atol=1e-9)


def test_future_ego_positions_empty() -> None:
    positions = future_ego_positions(EGO_AT_10_20_FACING_LEFT, [])
    assert positions.shape == (0, 2)
