# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Name-keyed registries of built-in IsaacTeleopCfg builders, for scripts that select a device by
CLI name (e.g. ``--teleop_device keyboard``) rather than reading an env-declared pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from .gamepad import se2_gamepad_teleop_cfg, se3_gamepad_teleop_cfg
from .keyboard import se2_keyboard_teleop_cfg, se3_keyboard_teleop_cfg
from .spacemouse import se2_spacemouse_teleop_cfg, se3_spacemouse_teleop_cfg

#: Device name -> builder returning an IsaacTeleopCfg for SE(3) delta-pose control (position +
#: rotation + optional gripper). Builders accept ``pos_sensitivity``, ``rot_sensitivity``, and
#: ``sim_device`` keyword arguments at minimum.
SE3_TELEOP_CFG_BUILDERS: dict[str, Callable[..., object]] = {
    "keyboard": se3_keyboard_teleop_cfg,
    "gamepad": se3_gamepad_teleop_cfg,
    "spacemouse": se3_spacemouse_teleop_cfg,
}

#: Device name -> builder returning an IsaacTeleopCfg for SE(2) base-velocity control. Builders
#: accept ``v_x_sensitivity``, ``v_y_sensitivity``, ``omega_z_sensitivity``, and ``sim_device``
#: keyword arguments at minimum.
SE2_TELEOP_CFG_BUILDERS: dict[str, Callable[..., object]] = {
    "keyboard": se2_keyboard_teleop_cfg,
    "gamepad": se2_gamepad_teleop_cfg,
    "spacemouse": se2_spacemouse_teleop_cfg,
}
