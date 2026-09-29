# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`~isaaclab_teleop.teleop_input.create_teleop_input`, against fake devices."""

from __future__ import annotations

import warnings

import isaaclab_teleop.teleop_input as teleop_input
import numpy as np
import pytest
from isaaclab_teleop.teleop_input import create_teleop_input

import isaaclab.utils
from isaaclab.devices import Se3KeyboardCfg, Se3SpaceMouseCfg

pytestmark = pytest.mark.unit

KEY_N, KEY_R = 49, 19


class _Group:
    """Stand-in for an OptionalTensorGroup in a pipeline step result."""

    def __init__(self, value):
        self._value = value
        self.is_none = False

    def __getitem__(self, index):
        return self._value


class _FakePipelineDevice:
    """Stand-in for an IsaacTeleopDevice: ``advance()`` returns the next queued step."""

    def __init__(self):
        self.steps: list[dict] = []
        self.last_step_result = None
        self.entered = self.exited = self.started = False
        self.kwargs: dict = {}

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def advance(self):
        step = self.steps.pop(0) if self.steps else None
        self._pressed = list(step.pop("pressed", [])) if step is not None else []
        self.last_step_result = step
        return "command" if step is not None else None

    def drain_pressed_keys(self):
        pressed, self._pressed = getattr(self, "_pressed", []), []
        return pressed

    def request_start(self):
        self.started = True

    def request_stop(self):
        pass

    def reset(self, pause=False):
        pass


def _keys(*codes):
    """A step during which ``codes`` were pressed on the input surfaces."""
    return {"pressed": list(codes)}


def _right_button():
    return {"spacemouse_buttons": _Group(np.array([0, 1]))}


@pytest.fixture
def pipeline_device(monkeypatch):
    # The pipeline path needs Isaac Capture (its key table and session); the fallback tests do not.
    pytest.importorskip("isaacteleop")
    import isaaclab_teleop.isaac_teleop_device as isaac_teleop_device

    device = _FakePipelineDevice()
    monkeypatch.setattr(teleop_input, "is_isaacteleop_available", lambda: True)

    def create(pipeline, **kwargs):
        device.kwargs = kwargs
        return device

    monkeypatch.setattr(isaac_teleop_device, "create_isaac_teleop_device", create)
    return device


class TestPipelineInput:
    def test_runs_headset_free_on_the_standalone_cloudxr_profile(self, pipeline_device):
        from isaaclab_teleop import CLOUDXR_STANDALONE_ENV

        with create_teleop_input(Se3KeyboardCfg()):
            assert pipeline_device.kwargs["cloudxr_env_file"] == CLOUDXR_STANDALONE_ENV
            assert pipeline_device.started

    def test_advance_returns_the_command_and_fires_key_callbacks(self, pipeline_device):
        fired = []
        with create_teleop_input(Se3KeyboardCfg()) as keyboard:
            keyboard.add_callback("N", lambda: fired.append("N"))
            pipeline_device.steps = [_keys(KEY_N)]

            assert keyboard.advance() == "command"
            assert keyboard.advance() is None  # session still starting: no command, no callbacks

        assert fired == ["N"]
        assert pipeline_device.entered and pipeline_device.exited

    def test_r_fires_on_the_key_and_the_spacemouse_right_button(self, pipeline_device):
        fired = []
        spacemouse = create_teleop_input(Se3SpaceMouseCfg())
        spacemouse.add_callback("R", lambda: fired.append("R"))
        pipeline_device.steps = [_keys(KEY_R), _right_button()]

        spacemouse.advance()
        spacemouse.advance()

        assert fired == ["R", "R"]


class TestDeprecatedDeviceInput:
    def test_without_isaacteleop_runs_the_deprecated_device_quietly(self, monkeypatch, mocker):
        legacy = mocker.MagicMock()
        legacy.advance.return_value = "legacy command"

        def instantiate(cfg):
            warnings.warn("Se3Keyboard is deprecated.", DeprecationWarning, stacklevel=2)
            return legacy

        monkeypatch.setattr(teleop_input, "is_isaacteleop_available", lambda: False)
        monkeypatch.setattr(isaaclab.utils, "instantiate", instantiate)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            keyboard = create_teleop_input(Se3KeyboardCfg())
        callback = mocker.MagicMock()
        keyboard.add_callback("N", callback)

        assert keyboard.advance() == "legacy command"
        legacy.add_callback.assert_called_once_with("N", callback)
