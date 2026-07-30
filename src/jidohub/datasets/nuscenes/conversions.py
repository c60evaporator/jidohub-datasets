"""nuScenes と標準スキーマの間の変換（純粋関数）。

**このモジュールは nuscenes-devkit に依存しない。** devkit のレコードから取り出した
プリミティブ（list / float / np.ndarray）だけを受け取り、標準スキーマの型を返す。

この分離の理由
    nuScenes と標準スキーマの差異（寸法順、座標系、速度、点群の形状）は
    **取り違えても例外が出ない**。検出できるのは既知の入出力を検証するテストだけであり、
    そのテストが devkit と 4GB のデータセットを要求するようでは CI で回せない。
    純粋関数に隔離することで、手計算で検証できるテストが書けるようになる。

    :mod:`jidohub.datasets.nuscenes.adapter` は devkit からの読み出しに徹し、
    計算式をここ以外に書かないこと（CLAUDE.md 2.1）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jidohub.core.geometry import invert_transform, quaternion_to_rotation_matrix
from jidohub.core.schemas import Box3D, DrivingCommand

__all__ = [
    "LIDAR_FIELDS",
    "transform_from_pose",
    "box_from_annotation",
    "points_from_devkit",
    "driving_command_from_future_positions",
]

LIDAR_FIELDS: tuple[str, ...] = ("x", "y", "z", "intensity", "ring")
"""nuScenes の LiDAR 点群の列名。devkit は ``(5, N)`` で返す。"""

_TURN_THRESHOLD_M = 2.0
"""走行指令を左折 / 右折と判定する横方向変位のしきい値[m]。

UniAD 系の実装で慣例的に使われる値。データセットに正解ラベルは存在せず、
将来の自車位置から**推定**するものである点に注意（:func:`driving_command_from_future_positions`）。
"""


def transform_from_pose(
    translation: Sequence[float] | np.ndarray,
    rotation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """``translation`` + quaternion の組から 4x4 同次変換行列を組み立てる。

    nuScenes の ``ego_pose`` と ``calibrated_sensor`` はどちらもこの形式で、
    **その座標系の点を親座標系へ移す**向きを表す。

    - ``ego_pose`` → ego → global（``Sample.ego_to_global``）
    - ``calibrated_sensor`` → sensor → ego（``CameraFrame.sensor_to_ego``）

    Args:
        translation: shape ``(3,)``。並進[m]。
        rotation: shape ``(4,)``。``(w, x, y, z)`` 順のクォータニオン。

    Returns:
        shape ``(4, 4)``、``np.float64`` の同次変換行列。
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(np.asarray(rotation, dtype=np.float64))
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def box_from_annotation(
    translation_global: Sequence[float] | np.ndarray,
    size_wlh: Sequence[float] | np.ndarray,
    rotation_global: Sequence[float] | np.ndarray,
    label: str,
    ego_to_global: np.ndarray,
    velocity_global: Sequence[float] | np.ndarray | None = None,
    track_id: int | None = None,
    attributes: dict | None = None,
) -> Box3D:
    """nuScenes の ``sample_annotation`` を ego 座標系の :class:`Box3D` に変換する。

    差異の対応（CLAUDE.md 3 章）
        - **寸法順**: nuScenes は ``(width, length, height)``、標準は ``(length, width, height)``。
          :meth:`Box3D.from_dimensions` にキーワードで渡すことで取り違えを防ぐ
        - **座標系**: nuScenes は global、標準は ego。``inv(ego_to_global)`` を適用する
        - **速度**: global 座標系で与えられ、前後フレームが無い場合は ``NaN`` になる。
          回転成分のみ適用して ego 座標へ移し、``NaN`` は ``None`` に落とす

    Args:
        translation_global: shape ``(3,)``。global 座標系でのボックス中心（幾何中心）[m]。
        size_wlh: shape ``(3,)``。**nuScenes の並び** ``(width, length, height)``[m]。
        rotation_global: shape ``(4,)``。global 座標系での ``(w, x, y, z)`` クォータニオン。
        label: クラス名（例: ``"vehicle.car"``）。カテゴリ名の正規化は呼び出し側の責務。
        ego_to_global: shape ``(4, 4)``。:func:`transform_from_pose` で作った変換。
        velocity_global: shape ``(3,)``。global 座標系での速度[m/s]。``None`` 可。
        track_id: 追跡 ID。``instance_token`` から採番する場合は呼び出し側で整数化する。
        attributes: データセット固有の属性。

    Returns:
        **ego 座標系**の :class:`Box3D`。
    """
    global_to_ego = invert_transform(ego_to_global)
    rotation_matrix = global_to_ego[:3, :3]

    center_ego = rotation_matrix @ np.asarray(translation_global, dtype=np.float64)
    center_ego += global_to_ego[:3, 3]

    # 姿勢も ego 座標系へ回す。回転行列の合成をクォータニオンに戻さず、
    # Box3D.rotation にはクォータニオンが必要なため行列から変換する
    orientation_global = quaternion_to_rotation_matrix(
        np.asarray(rotation_global, dtype=np.float64)
    )
    rotation_ego = _quaternion_from_rotation_matrix(rotation_matrix @ orientation_global)

    velocity_ego: np.ndarray | None = None
    if velocity_global is not None:
        velocity = np.asarray(velocity_global, dtype=np.float64)
        # 速度が NaN になるのは前後のキーフレームが存在しない場合。
        # 0 で埋めると「静止している」という誤った情報になるため None にする
        if not np.isnan(velocity).any():
            velocity_ego = rotation_matrix @ velocity

    width, length, height = (float(value) for value in size_wlh)
    return Box3D.from_dimensions(
        center=center_ego,
        length=length,  # nuScenes の size[1]
        width=width,  # nuScenes の size[0]
        height=height,  # nuScenes の size[2]
        rotation=rotation_ego,
        label=label,
        velocity=velocity_ego,
        track_id=track_id,
        attributes=attributes or {},
    )


def points_from_devkit(points: np.ndarray) -> np.ndarray:
    """devkit の ``LidarPointCloud.points`` を標準スキーマの形状に変換する。

    devkit は列優先の ``(C, N)`` で返すが、標準スキーマは ``(N, C)``。
    **転置を忘れると shape (5, N) が「5 点の点群」として通ってしまう**ため、
    列数が想定外に多い場合はエラーにする。

    Args:
        points: shape ``(C, N)``。devkit の ``LidarPointCloud.points``。

    Returns:
        shape ``(N, C)``、``np.float32`` の C 連続配列。
    """
    array = np.asarray(points)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array from the devkit, got shape {array.shape}")
    if array.shape[0] > array.shape[1]:
        raise ValueError(
            f"expected devkit layout (C, N) with C << N, got shape {array.shape}. "
            "The array may already be transposed."
        )
    return np.ascontiguousarray(array.T, dtype=np.float32)


def driving_command_from_future_positions(
    future_positions_ego: np.ndarray,
    threshold_m: float = _TURN_THRESHOLD_M,
) -> DrivingCommand:
    """将来の自車位置から高レベル走行指令を**推定**する。

    nuScenes に走行指令の正解ラベルは存在しないため、UniAD 系の実装に倣い
    将来の横方向変位から推定する。**推定値であることを利用側に伝えること**
    （評価に用いる場合、この定義の違いが結果に影響する）。

    Args:
        future_positions_ego: shape ``(T, 2)`` 以上。**現在の ego 座標系**での
            将来の自車位置[m]。``y`` が左方向正。
        threshold_m: 左折 / 右折と判定する横方向変位のしきい値[m]。

    Returns:
        推定された :class:`~jidohub.core.schemas.DrivingCommand`。
    """
    if future_positions_ego.ndim != 2 or future_positions_ego.shape[1] < 2:
        raise ValueError(
            f"future_positions_ego must have shape (T, >=2), got {future_positions_ego.shape}"
        )
    if len(future_positions_ego) == 0:
        return DrivingCommand.GO_STRAIGHT

    lateral = float(future_positions_ego[-1, 1])
    if lateral > threshold_m:
        return DrivingCommand.TURN_LEFT
    if lateral < -threshold_m:
        return DrivingCommand.TURN_RIGHT
    return DrivingCommand.GO_STRAIGHT


def _quaternion_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """3x3 回転行列を ``(w, x, y, z)`` のクォータニオンに変換する。

    数値的に安定な Shepperd 法（対角成分の最大値で場合分けする）を用いる。
    素朴な実装は ``w`` が 0 に近い場合に精度を失うため。
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)
