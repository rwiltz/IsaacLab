# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gamepad :class:`~isaaclab_teleop.IsaacTeleopCfg` builder functions.

Gamepad has no physical button wired to session start/stop/reset (those come from the
teleop session's control channel or an auxiliary control device), so unlike keyboard and
spacemouse there is no gamepad-specific control poller to test. These tests just verify the
builder functions wire up a correctly-shaped, gamepad-plugin-backed IsaacTeleopCfg.
"""

from __future__ import annotations

import sys
import types

# ``isaacteleop.plugins`` resolves an on-disk plugin search directory, which is irrelevant to
# these pure config-shape assertions and may not be present in every install of isaacteleop
# (e.g. minimal/CI builds). Stub it once at import time instead of per-test, so repeated
# construction of gamepad IsaacTeleopCfg objects below does not repeatedly patch and unpatch
# ``sys.modules``, which can trip unrelated import-machinery issues in third-party packages.
if "isaacteleop.plugins" not in sys.modules:
    _fake_plugins = types.ModuleType("isaacteleop.plugins")
    _fake_plugins.plugin_search_path = lambda: "/dummy/plugin/path"
    sys.modules["isaacteleop.plugins"] = _fake_plugins

from isaaclab_teleop.gamepad.se2_gamepad import se2_gamepad_teleop_cfg  # noqa: E402
from isaaclab_teleop.gamepad.se3_gamepad import se3_gamepad_teleop_cfg  # noqa: E402


class TestSe3GamepadTeleopCfg:
    def test_defaults(self):
        cfg = se3_gamepad_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabGamepadSe3"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "gamepad"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_gamepad_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"


class TestSe2GamepadTeleopCfg:
    def test_defaults(self):
        cfg = se2_gamepad_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabGamepadSe2"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "gamepad"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_gamepad_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
