# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse-driven IsaacTeleopCfg builder for SE(2) control."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..isaac_teleop_cfg import IsaacTeleopCfg


def se2_spacemouse_teleop_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    r"""Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for spacemouse-driven SE(2) velocity control.

    Raw axis/button state is retargeted via
    :class:`~isaacteleop.retargeters.SpaceMouseToSe2Retargeter`.

    The SpaceMouse is read in process from its HID device (``/dev/hidrawN``), so the pipeline
    needs no plugin process or headset. Reading ``/dev/hidraw*`` needs a udev rule
    granting the user access (see :ref:`SpaceMouse permissions <isaac-teleop-spacemouse-permissions>`).

    The interface finds and uses the first supported device connected to the computer.
    Currently tested for the SpaceMouse Compact
    (https://3dconnexion.com/de/product/spacemouse-compact/).

    The resulting pipeline's ``OutputCombiner`` also exposes a ``"spacemouse_buttons"`` output --
    poll it with :class:`~isaaclab_teleop.control_pollers.SpaceMouseResetPoller` for the right
    button's device-intrinsic reset, matching the legacy ``Se2SpaceMouse``.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from ..isaac_teleop_cfg import IsaacTeleopCfg

    def build_pipeline():
        from isaacteleop.retargeters import (
            SpaceMouseToSe2Retargeter,
            SpaceMouseToSe2RetargeterConfig,
            TensorReorderer,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        spacemouse_source = SpaceMouseSource("spacemouse")

        se2 = SpaceMouseToSe2Retargeter(
            SpaceMouseToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
            ),
            name="se2",
        )
        connected_se2 = se2.connect(
            {
                "spacemouse_translation": spacemouse_source.output("spacemouse_translation"),
                "spacemouse_rotation": spacemouse_source.output("spacemouse_rotation"),
            }
        )

        base_command_elements = ["v_x", "v_y", "omega_z"]
        reorderer = TensorReorderer(
            input_config={"base_command": base_command_elements},
            output_order=base_command_elements,
            name="action_reorderer",
            input_types={"base_command": "array"},
        )
        connected_reorderer = reorderer.connect({"base_command": connected_se2.output("base_command")})

        return OutputCombiner(
            {
                "action": connected_reorderer.output("output"),
                "spacemouse_buttons": spacemouse_source.output("spacemouse_buttons"),
            }
        )

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabSpaceMouseSe2",
    )
