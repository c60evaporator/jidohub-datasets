"""``NuScenesAdapter`` のテスト。

**nuScenes 本体（4GB）も devkit も使わない。** devkit の ``NuScenes`` / ``NuScenesCanBus``
は ``__init__`` に注入する設計なので、必要なレコードだけを返す stub で検証する。
点群読み込み（devkit 依存）はモジュール関数 ``read_lidar_points`` を差し替える。
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from jidohub.core.geometry import yaw_to_quaternion
from jidohub.core.schemas import CameraFrame, CoordinateFrame, DrivingCommand, Image
from PIL import Image as PILImage

from jidohub.datasets.nuscenes import adapter as adapter_module
from jidohub.datasets.nuscenes.adapter import NuScenesAdapter, read_image_bytes
from jidohub.datasets.nuscenes.conversions import intrinsic_from_devkit, points_from_pcd_bin

IMAGE_W, IMAGE_H = 32, 24
T0, T1, T2 = 1_000_000, 1_500_000, 2_000_000

# LiDAR 生データ: .pcd.bin のフラット float32 配列（8 点 × 5 列）。
# read_lidar_points が返すのと同じ形式で、センサ座標系のまま保持されることの検証に使う。
LIDAR_FLAT = np.arange(8 * 5, dtype=np.float32) * 0.1


class FakeNuScenes:
    """必要なレコードだけを返す最小の偽 devkit。"""

    def __init__(self, tables: dict[str, dict[str, dict]], paths: dict[str, str]) -> None:
        self._tables = tables
        self._paths = paths
        self.sample = list(tables["sample"].values())
        self.scene = list(tables["scene"].values())

    def get(self, table: str, token: str) -> dict:
        return self._tables[table][token]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        return self._paths[sample_data_token]

    def box_velocity(self, annotation_token: str) -> np.ndarray:
        return np.asarray(self._tables["_velocity"][annotation_token], dtype=np.float64)


class FakeCanBus:
    """``get_messages(scene_name, message_name)`` だけを持つ偽 CAN bus。"""

    def __init__(self, messages: dict[tuple[str, str], list[dict]]) -> None:
        self._messages = messages

    def get_messages(self, scene_name: str, message_name: str) -> list[dict]:
        return self._messages[(scene_name, message_name)]  # 該当が無ければ KeyError


def _ego_pose(token: str, x: float, y: float, yaw: float) -> dict:
    return {"token": token, "translation": [x, y, 0.0], "rotation": list(yaw_to_quaternion(yaw))}


def _make_jpeg(path: str) -> None:
    PILImage.new("RGB", (IMAGE_W, IMAGE_H), (120, 60, 200)).save(path, format="JPEG")


def build_dataset(tmp_path, *, ego_xy: list[tuple[float, float]]) -> FakeNuScenes:
    """1 シーン 3 サンプル・カメラ 1 枚の偽データセットを作る。

    ``ego_xy`` は各サンプルの LIDAR_TOP ego 位置（global）。自車は全サンプルで +y 向き。
    """
    yaw = math.pi / 2  # 全サンプル +y 向き
    sample_tokens = ["s0", "s1", "s2"]
    timestamps = [T0, T1, T2]

    scenes = {
        "scene0": {
            "token": "scene0",
            "name": "scene-0001",
            "first_sample_token": "s0",
        }
    }
    samples: dict[str, dict] = {}
    sample_data: dict[str, dict] = {}
    ego_poses: dict[str, dict] = {}
    annotations: dict[str, dict] = {}
    velocities: dict[str, list] = {}
    paths: dict[str, str] = {}

    # センサ校正（非単位回転。点群がセンサ座標系のまま保持されることの検証用）。
    calibrated = {
        "cs_lidar": {
            "token": "cs_lidar",
            "translation": [0.9, 0.0, 1.8],
            "rotation": list(yaw_to_quaternion(0.3)),
            "camera_intrinsic": [],
        },
        "cs_cam": {
            "token": "cs_cam",
            "translation": [1.0, 0.0, 1.5],
            "rotation": list(yaw_to_quaternion(-0.2)),
            "camera_intrinsic": [[1000.0, 0.0, 16.0], [0.0, 1000.0, 12.0], [0.0, 0.0, 1.0]],
        },
    }
    attributes = {"attr_moving": {"token": "attr_moving", "name": "vehicle.moving"}}

    instances = ["inst_A", "inst_A", "inst_B"]  # s0/s1 は同一物体、s2 は別物体
    labels = ["vehicle.car", "vehicle.car", "vehicle.truck"]

    for i, token in enumerate(sample_tokens):
        x, y = ego_xy[i]
        ego_poses[f"ep_{i}"] = _ego_pose(f"ep_{i}", x, y, yaw)

        sd_lidar = f"sd_lidar_{i}"
        sd_cam = f"sd_cam_{i}"
        sample_data[sd_lidar] = {
            "token": sd_lidar,
            "channel": "LIDAR_TOP",
            "calibrated_sensor_token": "cs_lidar",
            "ego_pose_token": f"ep_{i}",
            "height": 0,
            "width": 0,
            "timestamp": timestamps[i],
        }
        image_path = str(tmp_path / f"cam_{i}.jpg")
        _make_jpeg(image_path)
        paths[sd_cam] = image_path
        paths[sd_lidar] = str(tmp_path / f"lidar_{i}.bin")  # 中身は read_lidar_points で差し替え
        sample_data[sd_cam] = {
            "token": sd_cam,
            "channel": "CAM_FRONT",
            "calibrated_sensor_token": "cs_cam",
            "ego_pose_token": f"ep_{i}",
            "height": IMAGE_H,
            "width": IMAGE_W,
            "timestamp": timestamps[i],
        }

        ann_token = f"ann{i}"
        annotations[ann_token] = {
            "token": ann_token,
            "instance_token": instances[i],
            "translation": [x, y + 5.0, 0.0],  # 自車の 5m 前方
            "size": [1.9, 4.5, 1.6],  # nuScenes 順 (width, length, height)
            "rotation": list(yaw_to_quaternion(yaw)),
            "category_name": labels[i],
            "attribute_tokens": ["attr_moving"] if i == 0 else [],
        }
        velocities[ann_token] = [0.0, 5.0, 0.0]  # global 速度

        samples[token] = {
            "token": token,
            "timestamp": timestamps[i],
            "prev": sample_tokens[i - 1] if i > 0 else "",
            "next": sample_tokens[i + 1] if i < len(sample_tokens) - 1 else "",
            "scene_token": "scene0",
            "data": {"LIDAR_TOP": sd_lidar, "CAM_FRONT": sd_cam},
            "anns": [ann_token],
        }

    tables = {
        "scene": scenes,
        "sample": samples,
        "sample_data": sample_data,
        "ego_pose": ego_poses,
        "calibrated_sensor": calibrated,
        "sample_annotation": annotations,
        "attribute": attributes,
        "_velocity": velocities,
    }
    return FakeNuScenes(tables, paths)


@pytest.fixture(autouse=True)
def _patch_lidar(monkeypatch):
    """devkit の点群読み込みを差し替える（実データも devkit も不要にする）。"""
    monkeypatch.setattr(adapter_module, "read_lidar_points", lambda path: LIDAR_FLAT.copy())


@pytest.fixture
def straight_nusc(tmp_path):
    return build_dataset(tmp_path, ego_xy=[(10.0, 20.0), (10.0, 30.0), (10.0, 40.0)])


# ---------------------------------------------------------------------------
# ego_to_global / カメラ / 点群
# ---------------------------------------------------------------------------


def test_ego_to_global_from_lidar_ego_pose(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    sample = adapter.get_sample("s0")
    # LIDAR_TOP の ego_pose: global (10,20)、90° 左向き
    expected = np.eye(4)
    expected[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected[:3, 3] = [10.0, 20.0, 0.0]
    np.testing.assert_allclose(sample.ego_to_global, expected, atol=1e-9)


def test_camera_default_encoded(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    frame = adapter.get_sample("s0").cameras["CAM_FRONT"]
    # core 0.2: 画素・サイズ・intrinsic は Image が保持する（CameraFrame は持たない）。
    assert frame.image.is_encoded is True
    assert frame.image.height == IMAGE_H
    assert frame.image.width == IMAGE_W


def test_camera_pixels_mode(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc, image_mode="pixels")
    frame = adapter.get_sample("s0").cameras["CAM_FRONT"]
    assert frame.image.is_encoded is False
    assert frame.image.pixels is not None
    assert frame.image.pixels.shape == (IMAGE_H, IMAGE_W, 3)
    assert frame.image.height == IMAGE_H
    assert frame.image.width == IMAGE_W


def test_camera_intrinsic_carried_on_image(straight_nusc) -> None:
    """intrinsic が Image へ正しく引き継がれること（Image 移設時の取り違え検出）。"""
    adapter = NuScenesAdapter(straight_nusc)
    frame = adapter.get_sample("s0").cameras["CAM_FRONT"]
    # fixture の cs_cam.camera_intrinsic と一致すること。
    expected = intrinsic_from_devkit([[1000.0, 0.0, 16.0], [0.0, 1000.0, 12.0], [0.0, 0.0, 1.0]])
    assert frame.image.intrinsic is not None
    np.testing.assert_allclose(frame.image.intrinsic, expected)


def test_camera_frame_requires_intrinsic() -> None:
    """intrinsic を持たない Image を CameraFrame に渡すと ValueError（core 契約の確認）。

    Adapter が誤って intrinsic を落とすと、この契約に引っかかって即座に検出できる。
    """
    image = Image(pixels=np.zeros((2, 2, 3), dtype=np.uint8))  # intrinsic=None
    with pytest.raises(ValueError):
        CameraFrame(image=image, sensor_to_ego=np.eye(4), channel="CAM_FRONT")


def test_lidar_points_kept_in_sensor_frame(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    lidar = adapter.get_sample("s0").lidar
    assert lidar is not None
    # 点群は (N, 5)（ring_index を含む）、センサ座標系のまま（ego へ変換しない）。
    expected = points_from_pcd_bin(LIDAR_FLAT.copy())
    np.testing.assert_allclose(lidar.points, expected)
    assert lidar.points.shape == (8, 5)
    assert lidar.fields == ("x", "y", "z", "intensity", "ring")
    # sensor_to_ego が非単位であることを確認したうえで、points が未変換であること。
    assert not np.allclose(lidar.sensor_to_ego, np.eye(4))
    np.testing.assert_allclose(lidar.points, expected)


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_ascending_and_shallow(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc, history_length=2)
    sample = adapter.get_sample("s2")
    assert len(sample.history) == 2
    # 時刻昇順（末尾が直近の過去）
    assert [h.timestamp for h in sample.history] == [T0, T1]
    # history の Sample はさらに history を持たない（1 段のみ）
    assert all(h.history == [] for h in sample.history)


def test_history_at_scene_start_is_short_not_failing(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc, history_length=2)
    sample = adapter.get_sample("s0")  # prev が無い
    assert sample.history == []


# ---------------------------------------------------------------------------
# ground truth / scene 順序 / track_id
# ---------------------------------------------------------------------------


def test_ground_truth_in_ego_frame(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    gt = adapter.get_ground_truth("s0")
    assert gt.frame == CoordinateFrame.EGO
    assert len(gt.boxes) == 1
    box = gt.boxes[0]
    np.testing.assert_allclose(box.center, [5.0, 0.0, 0.0], atol=1e-9)
    assert box.label == "vehicle.car"
    assert box.length > box.width


def test_samples_in_scene_follows_linked_list(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    assert adapter.samples_in_scene("scene0") == ["s0", "s1", "s2"]
    assert adapter.sample_ids == ["s0", "s1", "s2"]
    assert adapter.scene_ids() == ["scene0"]


def test_track_id_consistent_within_scene(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)
    id_s0 = adapter.get_ground_truth("s0").boxes[0].track_id
    id_s1 = adapter.get_ground_truth("s1").boxes[0].track_id
    id_s2 = adapter.get_ground_truth("s2").boxes[0].track_id
    # s0/s1 は同一 instance → 同じ整数、s2 は別物体 → 異なる整数
    assert id_s0 == id_s1
    assert id_s2 != id_s0
    assert isinstance(id_s0, int)


# ---------------------------------------------------------------------------
# ego_state（CAN bus）
# ---------------------------------------------------------------------------


def _pose_msg(utime: int, vel, accel, rotation_rate) -> dict:
    return {"utime": utime, "vel": vel, "accel": accel, "rotation_rate": rotation_rate}


def test_ego_state_from_can_bus(straight_nusc) -> None:
    can_bus = FakeCanBus(
        {
            ("scene-0001", "pose"): [
                _pose_msg(T0, [3.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.2]),
            ],
            ("scene-0001", "vehicle_monitor"): [{"utime": T0, "steering": 90.0}],
        }
    )
    adapter = NuScenesAdapter(straight_nusc, can_bus=can_bus)
    state = adapter.get_sample("s0").ego_state
    assert state is not None
    np.testing.assert_allclose(state.velocity, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(state.angular_velocity, [0.0, 0.0, 0.2])
    assert state.steering_angle == pytest.approx(math.pi / 2)  # 度 → ラジアン


def test_ego_state_none_when_messages_far(straight_nusc) -> None:
    # 全メッセージが基準時刻から 10 秒離れている → None
    can_bus = FakeCanBus(
        {
            ("scene-0001", "pose"): [
                _pose_msg(T0 + 10_000_000, [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ],
            ("scene-0001", "vehicle_monitor"): [{"utime": T0 + 10_000_000, "steering": 0.0}],
        }
    )
    adapter = NuScenesAdapter(straight_nusc, can_bus=can_bus)
    assert adapter.get_sample("s0").ego_state is None


def test_ego_state_none_without_can_bus(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)  # can_bus 未指定
    assert adapter.get_sample("s0").ego_state is None


def test_ego_state_none_when_scene_missing_in_can_bus(straight_nusc) -> None:
    can_bus = FakeCanBus({})  # get_messages が KeyError を送出
    adapter = NuScenesAdapter(straight_nusc, can_bus=can_bus)
    # CAN bus データが無くても get_sample は失敗しない
    assert adapter.get_sample("s0").ego_state is None


# ---------------------------------------------------------------------------
# command（走行指令の推定）
# ---------------------------------------------------------------------------


def test_command_none_by_default(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc)  # command_horizon_s 未指定
    assert adapter.get_sample("s0").command is None


def test_command_go_straight(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc, command_horizon_s=2.0)
    assert adapter.get_sample("s0").command == DrivingCommand.GO_STRAIGHT


def test_command_turn_left(tmp_path) -> None:
    # s2 が自車の左（+x global 減少）へ大きく変位する経路
    nusc = build_dataset(tmp_path, ego_xy=[(10.0, 20.0), (8.0, 30.0), (4.0, 40.0)])
    adapter = NuScenesAdapter(nusc, command_horizon_s=2.0)
    assert adapter.get_sample("s0").command == DrivingCommand.TURN_LEFT


def test_command_none_at_scene_end(straight_nusc) -> None:
    adapter = NuScenesAdapter(straight_nusc, command_horizon_s=2.0)
    # 終端サンプルは将来フレームが無い → None（失敗しない）
    assert adapter.get_sample("s2").command is None


# ---------------------------------------------------------------------------
# read_image_bytes（マジックバイト判定）
# ---------------------------------------------------------------------------


def test_read_image_bytes_detects_jpeg(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    _make_jpeg(str(path))
    data, image_format = read_image_bytes(path)
    assert image_format.value == "jpeg"
    assert data.startswith(b"\xff\xd8\xff")


def test_read_image_bytes_detects_png(tmp_path) -> None:
    buffer = io.BytesIO()
    PILImage.new("RGB", (4, 4)).save(buffer, format="PNG")
    path = tmp_path / "a.png"
    path.write_bytes(buffer.getvalue())
    _data, image_format = read_image_bytes(path)
    assert image_format.value == "png"


def test_read_image_bytes_rejects_unknown(tmp_path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"not an image at all")
    with pytest.raises(ValueError):
        read_image_bytes(path)


# ---------------------------------------------------------------------------
# その他のテスト
# ---------------------------------------------------------------------------
def test_adapter_is_importable_from_package_root() -> None:
    """README に記載の import 経路が有効であること（devkit 不要）。"""
    from jidohub.datasets.nuscenes import NuScenesAdapter

    assert NuScenesAdapter is not None
