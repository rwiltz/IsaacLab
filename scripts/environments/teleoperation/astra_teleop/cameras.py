# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera configuration and copied image inputs for Astra inference."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import torch
from PIL import Image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def add_wrist_cameras(scene_cfg, camera_names: list[str]) -> None:
    """Attach selected RGB cameras to the wrist bodies, inheriting the resolved head renderer.

    These are virtual mounts, not a calibration of physical G1 cameras. Positions
    and XYZW rotations use each wrist link's local coordinates and the ROS camera
    convention (+Z forward, -Y up). Each looks back toward the grip opening at
    local (0.12, -/+0.035, 0), from beyond the fingertips on the back-of-hand side.
    This keeps the camera above the table when the fingers curl downward.
    """
    from isaaclab.sensors import CameraCfg

    # CameraCfg otherwise reports its initialization pose even while its parent moves.
    scene_cfg.robot_pov_cam.update_latest_camera_pose = True
    mounts = {
        "left": ((0.23, 0.02, 0.12), (-0.4844679, -0.7838854, 0.3303489, 0.2041668)),
        "right": ((0.23, -0.02, 0.12), (-0.7838855, -0.4844679, 0.2041668, 0.3303488)),
    }
    for side, (position, rotation) in mounts.items():
        name = f"{side}_wrist_cam"
        if name in camera_names:
            camera = scene_cfg.robot_pov_cam.replace(
                prim_path=f"{scene_cfg.robot.prim_path}/{side}_wrist_yaw_link/AstraWristCam",
                offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="ros"),
                width=640,
                height=480,
                spawn=scene_cfg.robot_pov_cam.spawn.replace(
                    focal_length=10.5, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
                ),
            )
            setattr(scene_cfg, name, camera)


def capture_rgb_images(env: ManagerBasedRLEnv, camera_names: list[str], max_size: int) -> list[dict[str, str]]:
    """Detach RGB camera buffers and limit their longest encoded edge to ``max_size``."""
    if max_size < 1:
        raise ValueError("Camera max_size must be positive.")
    images = []
    for name in camera_names:
        camera = env.scene.sensors[name]
        output = camera.data.output
        if output is None or "rgb" not in output:
            raise ValueError(f"Camera {name!r} has no RGB output. Enable camera rendering before launching the app.")
        pixels = output["rgb"].torch[0, ..., :3].detach().to(device="cpu", copy=True)
        if pixels.dtype != torch.uint8:
            if not torch.isfinite(pixels).all():
                raise ValueError(f"Camera {name!r} contains non-finite pixels.")
            if pixels.max() <= 1:
                pixels = pixels * 255
            pixels = pixels.clamp(0, 255).to(torch.uint8)
        frame = Image.fromarray(pixels.numpy())
        frame.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        images.append({"name": name, "data_url": f"data:image/jpeg;base64,{encoded}"})
    return images
