#!/usr/bin/env python3
"""nuScenes 実データでの疎通確認。

stub テストでは検証できない「実データの中身の妥当性」を確認する。
CI では動かさない（実データが必要なため）。開発者が手元で 1 度実行する。

使い方::

    python scripts/smoke_real_data.py --dataroot /data/nuscenes --version v1.0-mini
    python scripts/smoke_real_data.py --dataroot /data/nuscenes --version v1.0-mini --can-bus

検証の狙い
    寸法の並べ替え・座標系・変換行列の向きの誤りは**例外を出さない**。
    実データの統計と幾何的な整合性で間接的に検出する。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from jidohub.core.geometry import invert_transform
from jidohub.core.schemas import DrivingCommand

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((ok, name, detail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--can-bus", action="store_true", help="CAN bus 拡張も検証する")
    args = parser.parse_args()

    from nuscenes import NuScenes

    from jidohub.datasets.nuscenes import NuScenesAdapter

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    can_bus = None
    if args.can_bus:
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus

        can_bus = NuScenesCanBus(dataroot=args.dataroot)

    adapter = NuScenesAdapter(nusc, can_bus=can_bus, history_length=2, command_horizon_s=3.0)

    sample_ids = adapter.sample_ids[: args.num_samples]
    samples = [adapter.get_sample(sid) for sid in sample_ids]
    truths = [adapter.get_ground_truth(sid) for sid in sample_ids]

    _check_timestamps(samples)
    _check_images(samples)
    _check_lidar(samples)
    _check_history(samples)
    _check_box_dimensions(truths)
    _check_box_positions(truths)
    _check_velocity(truths)
    _check_projection(samples, truths)
    _check_command(samples)
    if can_bus is not None:
        _check_ego_state(samples)

    width = max(len(name) for _, name, _ in _results)
    failed = sum(1 for ok, _, _ in _results if not ok)
    for ok, name, detail in _results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print()
    print(f"{len(_results) - failed} passed, {failed} failed")
    return 1 if failed else 0


def _check_timestamps(samples) -> None:
    """時刻がマイクロ秒であること。秒や ns なら桁が変わる。"""
    ts = samples[0].timestamp
    # nuScenes は 2018 年頃の収録。μs なら 1.5e15 前後になる
    check("timestamp がマイクロ秒", 1e15 < ts < 2e15, f"{ts}")


def _check_images(samples) -> None:
    frames = samples[0].cameras
    check("カメラ 6 枚", len(frames) == 6, f"{sorted(frames)}")

    front = frames["CAM_FRONT"]
    check("既定で encoded", front.is_encoded, f"is_encoded={front.is_encoded}")
    check(
        "画像サイズ 1600x900",
        (front.width, front.height) == (1600, 900),
        f"{front.width}x{front.height}",
    )

    # デコードして宣言サイズと一致することを確認（デコーダの契約検証）
    pixels = front.image
    check(
        "デコード結果が宣言サイズと一致",
        pixels.shape == (front.height, front.width, 3) and pixels.dtype == np.uint8,
        f"{pixels.shape} {pixels.dtype}",
    )

    # RGB 順の間接確認: 空が写る上部は青が強いはず（BGR なら赤が強く出る）
    top = pixels[: pixels.shape[0] // 4].reshape(-1, 3).mean(axis=0)
    check(
        "RGB 順（上部で B > R）", top[2] > top[0], f"R={top[0]:.0f} G={top[1]:.0f} B={top[2]:.0f}"
    )


def _check_lidar(samples) -> None:
    sweep = samples[0].lidar
    check(
        "点群が (N, 5)",
        sweep.points.ndim == 2 and sweep.points.shape[1] == 5,
        f"{sweep.points.shape}",
    )
    check("点数が妥当（2〜5 万点）", 20_000 < len(sweep.points) < 50_000, f"{len(sweep.points)} 点")
    check("dtype が float32", sweep.points.dtype == np.float32, f"{sweep.points.dtype}")

    # ring_index は 0..31 の整数値（VLP-32）。転置ミスや列ズレならこの範囲を外れる
    ring = sweep.points[:, 4]
    check(
        "ring が 0〜31 の整数",
        ring.min() >= 0 and ring.max() <= 31 and np.allclose(ring, np.round(ring)),
        f"min={ring.min():.1f} max={ring.max():.1f}",
    )

    # intensity は 0..255
    intensity = sweep.points[:, 3]
    check(
        "intensity が 0〜255",
        0 <= intensity.min() and intensity.max() <= 255,
        f"min={intensity.min():.0f} max={intensity.max():.0f}",
    )

    # センサ座標系のまま = 原点付近に自車があるので、z が概ね負側に分布する
    xyz = sweep.points[:, :3]
    check(
        "点群がセンサ座標系（半径 100m 以内）",
        np.linalg.norm(xyz[:, :2], axis=1).max() < 120,
        f"最大水平距離 {np.linalg.norm(xyz[:, :2], axis=1).max():.1f}m",
    )


def _check_history(samples) -> None:
    with_history = [s for s in samples if s.history]
    if not with_history:
        check("history", False, "history を持つ Sample が無い")
        return
    sample = with_history[0]
    times = [h.timestamp for h in sample.history]
    check("history が時刻昇順", times == sorted(times), f"{len(times)} 件")
    check("history が現在より過去", max(times) < sample.timestamp, "")
    check("history が入れ子でない", all(not h.history for h in sample.history), "")


def _check_box_dimensions(truths) -> None:
    """寸法の並べ替えミスを統計で検出する。"""
    cars = [b for gt in truths for b in gt.boxes if b.label == "vehicle.car"]
    if not cars:
        check("車両の寸法", False, "vehicle.car が見つからない")
        return

    lengths = np.array([b.length for b in cars])
    widths = np.array([b.width for b in cars])
    heights = np.array([b.height for b in cars])

    check(
        "車長の中央値が 4〜5m", 4.0 < np.median(lengths) < 5.5, f"median={np.median(lengths):.2f}m"
    )
    check(
        "車幅の中央値が 1.6〜2.1m",
        1.6 < np.median(widths) < 2.1,
        f"median={np.median(widths):.2f}m",
    )
    check(
        "車高の中央値が 1.4〜2.0m",
        1.4 < np.median(heights) < 2.0,
        f"median={np.median(heights):.2f}m",
    )
    # 並べ替えを誤ると length < width になる
    check(
        "全車両で length > width",
        bool(np.all(lengths > widths)),
        f"{np.sum(lengths <= widths)} 件が違反",
    )


def _check_box_positions(truths) -> None:
    """ego 座標系への変換が効いていることを確認する。"""
    centers = np.array([b.center for gt in truths for b in gt.boxes])
    if len(centers) == 0:
        check("ボックス位置", False, "アノテーションが無い")
        return

    distances = np.linalg.norm(centers[:, :2], axis=1)
    # global 座標のままなら数百〜数千 m のオーダーになる
    check(
        "ボックスが ego 座標（半径 100m 以内）",
        distances.max() < 120,
        f"最大 {distances.max():.1f}m",
    )
    check("近傍にボックスが存在する", distances.min() < 30, f"最小 {distances.min():.1f}m")
    check(
        "z が地面付近（-3〜3m）",
        -3 < np.median(centers[:, 2]) < 3,
        f"median z={np.median(centers[:, 2]):.2f}m",
    )


def _check_velocity(truths) -> None:
    boxes = [b for gt in truths for b in gt.boxes]
    with_velocity = [b for b in boxes if b.velocity is not None]
    check(
        "速度が一部 None（NaN を 0 で埋めていない）",
        len(with_velocity) < len(boxes),
        f"{len(with_velocity)}/{len(boxes)} 件に速度あり",
    )
    if with_velocity:
        speeds = np.array([np.linalg.norm(b.velocity[:2]) for b in with_velocity])
        check("速度が現実的（最大 40m/s 未満）", speeds.max() < 40, f"最大 {speeds.max():.1f}m/s")


def _check_projection(samples, truths) -> None:
    """最重要: 変換行列の向きを幾何的に検証する。

    ego 座標のボックス中心をカメラ画像へ投影し、前方のボックスが
    CAM_FRONT の画角内に落ちることを確認する。``sensor_to_ego`` の向きが
    逆なら、この投影は成立しない。
    """
    hits = 0
    total = 0
    for sample, gt in zip(samples, truths):
        frame = sample.cameras.get("CAM_FRONT")
        if frame is None:
            continue
        ego_to_camera = invert_transform(frame.sensor_to_ego)
        for box in gt.boxes:
            # ego 座標で前方 5〜40m、横方向 ±5m のボックスに限定
            x, y = box.center[0], box.center[1]
            if not (5 < x < 40 and abs(y) < 5):
                continue
            total += 1
            point = ego_to_camera[:3, :3] @ box.center + ego_to_camera[:3, 3]
            if point[2] <= 0:  # カメラ後方
                continue
            uv = frame.intrinsic @ point
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            if 0 <= u < frame.width and 0 <= v < frame.height:
                hits += 1

    if total == 0:
        check("投影の整合性", False, "前方に対象ボックスが無い")
        return
    ratio = hits / total
    # 前方 ±5m のボックスはほぼ全て CAM_FRONT に写るはず
    check(
        "前方ボックスが CAM_FRONT に投影される（変換の向きの検証）",
        ratio > 0.8,
        f"{hits}/{total} = {ratio:.0%}",
    )


def _check_command(samples) -> None:
    commands = [s.command for s in samples]
    check(
        "command が導出されている",
        all(c is not None for c in commands[:-3]),
        f"{sum(c is not None for c in commands)}/{len(commands)} 件",
    )
    kinds = {c for c in commands if c is not None}
    check("command が妥当な値", kinds <= set(DrivingCommand), f"{sorted(c.value for c in kinds)}")


def _check_ego_state(samples) -> None:
    states = [s.ego_state for s in samples if s.ego_state is not None]
    check("ego_state が取得できている", len(states) > 0, f"{len(states)}/{len(samples)} 件")
    if not states:
        return
    speeds = np.array([np.linalg.norm(s.velocity[:2]) for s in states])
    check(
        "自車速度が現実的（0〜30m/s）",
        0 <= speeds.min() and speeds.max() < 30,
        f"{speeds.min():.1f}〜{speeds.max():.1f} m/s",
    )
    # ego 座標系なので前進時は x 成分が支配的
    forward = np.array([s.velocity[0] for s in states])
    lateral = np.array([abs(s.velocity[1]) for s in states])
    check(
        "速度が ego 座標系（前後成分が支配的）",
        np.median(np.abs(forward)) > np.median(lateral),
        f"median |vx|={np.median(np.abs(forward)):.2f} |vy|={np.median(lateral):.2f}",
    )
    steering = [s.steering_angle for s in states if s.steering_angle is not None]
    if steering:
        check(
            "ステアリング角がラジアン（|θ| < 12rad）",
            max(abs(a) for a in steering) < 12,
            f"最大 {max(abs(a) for a in steering):.2f} rad",
        )


if __name__ == "__main__":
    sys.exit(main())
