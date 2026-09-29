# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gamepad :class:`~isaaclab_teleop.IsaacTeleopCfg` builder functions.

Gamepad has no physical button wired to session start/stop/reset (those come from the
teleop session's control channel or the B/P/R keys), so unlike keyboard and
spacemouse there is no gamepad-specific control poller to test. These tests just verify the
builder functions wire up a correctly-shaped, plugin-free gamepad IsaacTeleopCfg.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.gamepad.se2_gamepad import se2_gamepad_teleop_cfg
from isaaclab_teleop.gamepad.se3_gamepad import se3_gamepad_teleop_cfg


def _has_gamepad_source(cfg) -> bool:
    """Whether the config's pipeline reads the gamepad in process."""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource, find_sources

    return len(find_sources(cfg.pipeline_builder(), GamepadSource)) == 1


class TestSe3GamepadTeleopCfg:
    def test_defaults(self):
        cfg = se3_gamepad_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabGamepadSe3"
        # read in process: no plugin
        assert cfg.plugins == []
        assert _has_gamepad_source(cfg)
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
        # read in process: no plugin
        assert cfg.plugins == []
        assert _has_gamepad_source(cfg)
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_gamepad_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
