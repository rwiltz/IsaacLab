# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve a script's teleop device request with an Isaac Capture pipeline when one can.

The keyboard, gamepad and SpaceMouse devices in :mod:`isaaclab.devices` are deprecated. Their
configs (:class:`~isaaclab.devices.Se3KeyboardCfg`, ...) still declare an environment's devices,
and :func:`select_isaac_teleop_cfg` translates them into the equivalent Isaac Capture pipeline.
Without the ``isaacteleop`` package, or for a device it has no pipeline for (e.g. OpenXR hand
tracking), it returns ``None`` and the caller keeps using the deprecated device.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from isaaclab.devices.gamepad import Se2GamepadCfg, Se3GamepadCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg, Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se2SpaceMouseCfg, Se3SpaceMouseCfg

from .gamepad import se2_gamepad_teleop_cfg, se3_gamepad_teleop_cfg
from .keyboard import se2_keyboard_teleop_cfg, se3_keyboard_teleop_cfg
from .spacemouse import se2_spacemouse_teleop_cfg, se3_spacemouse_teleop_cfg

if TYPE_CHECKING:
    from isaaclab.devices import DeviceCfg

    from .isaac_teleop_cfg import IsaacTeleopCfg


def is_isaacteleop_available() -> bool:
    """Whether the ``isaacteleop`` package, which runs Isaac Capture pipelines, is installed."""
    return importlib.util.find_spec("isaacteleop") is not None


def pipeline_from_device_cfg(device_cfg: DeviceCfg, sim_device: str) -> IsaacTeleopCfg | None:
    """Build the Isaac Capture pipeline equivalent to a device config.

    Args:
        device_cfg: A device config, e.g. :class:`~isaaclab.devices.Se3KeyboardCfg`.
        sim_device: Torch device for the pipeline's output tensors.

    Returns:
        The pipeline, or ``None`` when no pipeline serves this device config type.
    """
    cfg = device_cfg
    cfg_type = type(cfg)
    if cfg_type is Se3KeyboardCfg:
        return se3_keyboard_teleop_cfg(
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            gripper_term=cfg.gripper_term,
            sim_device=sim_device,
        )
    if cfg_type is Se2KeyboardCfg:
        return se2_keyboard_teleop_cfg(
            v_x_sensitivity=cfg.v_x_sensitivity,
            v_y_sensitivity=cfg.v_y_sensitivity,
            omega_z_sensitivity=cfg.omega_z_sensitivity,
            sim_device=sim_device,
        )
    if cfg_type is Se3GamepadCfg:
        return se3_gamepad_teleop_cfg(
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            dead_zone=cfg.dead_zone,
            gripper_term=cfg.gripper_term,
            sim_device=sim_device,
        )
    if cfg_type is Se2GamepadCfg:
        return se2_gamepad_teleop_cfg(
            v_x_sensitivity=cfg.v_x_sensitivity,
            v_y_sensitivity=cfg.v_y_sensitivity,
            omega_z_sensitivity=cfg.omega_z_sensitivity,
            dead_zone=cfg.dead_zone,
            sim_device=sim_device,
        )
    if cfg_type is Se3SpaceMouseCfg:
        return se3_spacemouse_teleop_cfg(
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            gripper_term=cfg.gripper_term,
            sim_device=sim_device,
        )
    if cfg_type is Se2SpaceMouseCfg:
        return se2_spacemouse_teleop_cfg(
            v_x_sensitivity=cfg.v_x_sensitivity,
            v_y_sensitivity=cfg.v_y_sensitivity,
            omega_z_sensitivity=cfg.omega_z_sensitivity,
            sim_device=sim_device,
        )
    return None


def select_isaac_teleop_cfg(
    env_cfg: Any,
    device_name: str | None,
    default_device_cfgs: Mapping[str, DeviceCfg],
    sim_device: str,
) -> IsaacTeleopCfg | None:
    """Pick the Isaac Capture pipeline that serves a script's device request.

    Without ``device_name``, the environment's declared ``isaac_teleop`` pipeline is used, else
    the ``"keyboard"`` entry of ``default_device_cfgs``. With it, the environment's
    ``teleop_devices`` entry of that name is used, else the ``default_device_cfgs`` entry.

    Args:
        env_cfg: The environment config, read for ``isaac_teleop`` and ``teleop_devices``.
        device_name: The requested device (e.g. ``--teleop_device``), or ``None``.
        default_device_cfgs: The script's own device configs, by name, for devices the
            environment does not declare.
        sim_device: Torch device for the pipeline's output tensors.

    Returns:
        The pipeline, or ``None`` when ``isaacteleop`` is not installed or no pipeline serves the
        requested device; the caller then uses the deprecated device path.
    """
    if not is_isaacteleop_available():
        return None
    if device_name is None:
        declared = getattr(env_cfg, "isaac_teleop", None)
        if declared is not None:
            return declared
        device_name = "keyboard"
    env_devices = getattr(getattr(env_cfg, "teleop_devices", None), "devices", None) or {}
    device_cfg = env_devices.get(device_name) or default_device_cfgs.get(device_name)
    if device_cfg is None:
        return None
    return pipeline_from_device_cfg(device_cfg, sim_device)
