# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven IsaacTeleopCfg builder for SE(2) control."""

from __future__ import annotations

# Tensor collection ID shared by the bundled keyboard plugin and its KeyboardTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_KEYBOARD_COLLECTION_ID = "keyboard"


def se2_keyboard_teleop_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
):
    r"""Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard-driven SE(2) velocity control.

    Raw key state is read through a bundled ``keyboard`` IsaacTeleop plugin (Linux evdev) and
    retargeted via :class:`~isaacteleop.retargeters.KeyboardToSe2Retargeter`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        ====================== ========================= ========================

    The resulting pipeline's ``OutputCombiner`` also exposes a ``"keyboard_all_keys"`` output --
    poll it with :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller` for physical
    B (start/resume) / P (pause) / R (reset) control-key handling; register an "L" callback on
    the same poller for a device-reset key, matching the legacy ``Se2Keyboard``.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    from ..isaac_teleop_cfg import IsaacTeleopCfg

    def build_pipeline():
        from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig, TensorReorderer
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        keyboard_source = KeyboardSource(_KEYBOARD_COLLECTION_ID)

        se2 = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
            ),
            name="se2",
        )
        connected_se2 = se2.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})

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
                "keyboard_all_keys": keyboard_source.output("keyboard_all_keys"),
            }
        )

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        plugins=[
            PluginConfig(
                plugin_name="keyboard",
                plugin_root_id=_KEYBOARD_COLLECTION_ID,
                search_paths=[plugin_search_path()],
            )
        ],
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe2",
    )
