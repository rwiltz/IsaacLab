# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Capture plugin that reads the SpaceMouse for the SpaceMouse teleop configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacteleop.teleop_session_manager import PluginConfig


def spacemouse_plugin_configs() -> list[PluginConfig]:
    """The Isaac Capture ``spacemouse`` plugin, started with the session from the copy bundled
    with Isaac Capture. It pushes to the collection ``SpaceMouseSource("spacemouse")`` reads."""
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    return [
        PluginConfig(
            plugin_name="spacemouse",
            plugin_root_id="spacemouse",
            search_paths=[BUNDLED_PLUGINS_DIR],
            required=True,
        )
    ]
