# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A keyboard, gamepad or SpaceMouse that a script drives directly, frame by frame."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from .device_selection import is_isaacteleop_available, pipeline_from_device_cfg

if TYPE_CHECKING:
    import torch

    from isaaclab.devices import DeviceCfg

    from .isaac_teleop_cfg import IsaacTeleopCfg

logger = logging.getLogger(__name__)


class TeleopInput(ABC):
    """A teleop device a script drives directly: a command per frame plus key callbacks.

    Call :meth:`advance` once per frame; callbacks fire from it. Use as a context manager, or
    call :meth:`close` when done.
    """

    @abstractmethod
    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        """Call ``func`` when ``key`` is pressed.

        Args:
            key: Key name, e.g. ``"N"``. ``"R"`` also fires on the SpaceMouse's right button.
            func: The function to call. Takes no arguments.
        """

    @abstractmethod
    def advance(self) -> torch.Tensor | None:
        """Read the device and fire the callbacks of keys pressed since the last call.

        Returns:
            The device's command, or ``None`` while it has none yet.
        """

    def close(self) -> None:
        """Release the device."""

    def __enter__(self) -> TeleopInput:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _PipelineInput(TeleopInput):
    """Runs the device through an Isaac Capture pipeline; keys come from the focused viewer window."""

    def __init__(self, pipeline: IsaacTeleopCfg) -> None:
        from .control_pollers import KeyboardControlPoller, SpaceMouseResetPoller
        from .isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV
        from .isaac_teleop_device import create_isaac_teleop_device

        # No headset: run on the standalone CloudXR profile and start locally, like the teleop
        # scripts without --xr.
        self._device = create_isaac_teleop_device(
            pipeline, use_kit_xr_bridge=False, cloudxr_env_file=CLOUDXR_STANDALONE_ENV
        )
        self._device.__enter__()
        self._device.request_start()
        self._keys = KeyboardControlPoller(self._device)
        self._right_button = SpaceMouseResetPoller(self._device, on_reset=self._on_right_button)
        self._on_reset: Callable[[], None] | None = None

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._keys.add_callback(key, func)
        if key.upper() == "R":
            self._on_reset = func

    def advance(self) -> torch.Tensor | None:
        command = self._device.advance()
        self._keys.advance()
        self._right_button.advance()
        return command

    def close(self) -> None:
        self._device.__exit__(None, None, None)

    def _on_right_button(self) -> None:
        if self._on_reset is not None:
            self._on_reset()


class _DeprecatedDeviceInput(TeleopInput):
    """Runs the deprecated :mod:`isaaclab.devices` device, for systems without ``isaacteleop``."""

    def __init__(self, device_cfg: DeviceCfg) -> None:
        from isaaclab.utils import instantiate

        logger.info("isaacteleop is not installed; using the deprecated %s", type(device_cfg).__name__)
        # the fallback is chosen here, not by the caller, so its deprecation warning is not theirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._device = instantiate(device_cfg)

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._device.add_callback(key, func)

    def advance(self) -> torch.Tensor | None:
        return self._device.advance()


def create_teleop_input(device_cfg: DeviceCfg) -> TeleopInput:
    """Create the device a device config describes, through Isaac Capture when it is installed.

    Args:
        device_cfg: A keyboard, gamepad or SpaceMouse config from :mod:`isaaclab.devices`, e.g.
            :class:`~isaaclab.devices.Se3KeyboardCfg`.

    Returns:
        The device. It runs as an Isaac Capture pipeline when ``isaacteleop`` is installed, and as
        the deprecated :mod:`isaaclab.devices` device otherwise.
    """
    pipeline = pipeline_from_device_cfg(device_cfg, device_cfg.sim_device) if is_isaacteleop_available() else None
    if pipeline is None:
        return _DeprecatedDeviceInput(device_cfg)
    return _PipelineInput(pipeline)
