# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect the generic adapter boundary without starting a simulator."""

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.utils.string import resolve_matching_names

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.adapters import CartpoleAdapter  # noqa: E402


class Scene(dict):
    """Named assets and sensors over detached CPU measurements."""


@pytest.fixture
def environment():
    # Deliberately reverse the common cart-then-pole order and use nonzero defaults.
    names = ["cart_to_pole", "slider_to_cart"]
    data = SimpleNamespace(
        joint_names=names,
        joint_pos=SimpleNamespace(torch=torch.tensor([[0.25, -1.5]])),
        joint_vel=SimpleNamespace(torch=torch.tensor([[-0.5, 2.0]])),
    )
    robot = SimpleNamespace(data=data, find_joints=lambda keys: resolve_matching_names(keys, names))
    term = SimpleNamespace(cfg=JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0))
    scene = Scene(robot=robot)
    scene.sensors = {}
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(episode_length_s=5.0),
        scene=scene,
        action_manager=SimpleNamespace(active_terms=["joint_effort"], action_term_dim=[1], get_term=lambda name: term),
        observation_manager=SimpleNamespace(active_terms={"policy": ["joint_pos_rel", "joint_vel_rel"]}),
    )


def test_named_state_and_normalized_action_survive_joint_order_and_buffer_reuse(environment):
    adapter = CartpoleAdapter(environment)
    observations = {"policy": torch.tensor([[0.125, -1.0, -0.25, 1.0]])}
    snapshot = adapter.snapshot(observations, episode_id=3, step=12, sim_time=0.2)
    assert snapshot["state"] == {
        "cart_position_m": -1.5,
        "pole_angle_rad": 0.25,
        "cart_velocity_m_s": 2.0,
        "pole_velocity_rad_s": -0.5,
    }
    assert snapshot["policy"] == [0.125, -1.0, -0.25, 1.0]
    assert adapter.describe()["policy_layout"][0] == "cart_to_pole.position_relative_to_default"
    environment.scene["robot"].data.joint_pos.torch.fill_(100)
    observations["policy"].fill_(100)
    assert snapshot["state"]["cart_position_m"] == -1.5
    assert snapshot["policy"][1] == -1.0
    json.dumps(snapshot, allow_nan=False)

    action = adapter.action({"cart_force": 0.25})
    assert action.shape == (1, 1) and action.dtype == torch.float32
    assert action.item() == 0.25  # The native action term applies the 100 N scale once.
    assert adapter.describe()["force_newtons_per_unit"] * action.item() == 25.0
    action.fill_(1)
    assert adapter.idle_action().item() == 0.0
    adapter.contract.validate_command(snapshot["neutral_command"])
    adapter.reset()
    assert adapter.task_progress(0.0) is None


def test_invalid_commands_and_incompatible_effort_mapping_are_rejected(environment):
    adapter = CartpoleAdapter(environment)
    for command in ({}, {"cart_force": True}, {"cart_force": float("nan")}, {"cart_force": 1e300}):
        with pytest.raises(ValueError):
            adapter.action(command)
    environment.action_manager.get_term("joint_effort").cfg.offset = 5.0
    with pytest.raises(ValueError, match="zero offset"):
        CartpoleAdapter(environment)
    environment.action_manager.get_term("joint_effort").cfg.offset = 0.0
    environment.action_manager.get_term("joint_effort").cfg.joint_names = ["cart_to_pole"]
    with pytest.raises(ValueError, match="slider_to_cart"):
        CartpoleAdapter(environment)
    environment.action_manager.action_term_dim = [2]
    with pytest.raises(ValueError, match="1D"):
        CartpoleAdapter(environment)


def test_optional_camera_copy_and_state_only_configuration_preserve_native_task(environment):
    adapter = CartpoleAdapter(environment)
    args = SimpleNamespace(cameras=[], enable_cameras=True, require_kit=False)
    CartpoleAdapter.configure_env(environment.cfg, args)
    assert not args.enable_cameras and not args.require_kit
    assert environment.cfg.episode_length_s == 5.0
    assert adapter.capture_images([], 10) == []

    pixels = torch.zeros(1, 20, 40, 4)
    pixels[..., 0] = 1.0
    environment.scene.sensors["view"] = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=pixels)})
    )
    images = adapter.capture_images(["view"], 10)
    pixels.zero_()
    assert images[0]["name"] == "view"
    frame = Image.open(io.BytesIO(base64.b64decode(images[0]["data_url"].split(",", 1)[1])))
    assert frame.size == (10, 5)
    red, green, blue = frame.getpixel((4, 2))
    assert red >= 253 and green <= 2 and blue <= 2
    environment.scene.sensors["view"].data.output = {"depth": None}
    with pytest.raises(ValueError, match="no RGB"):
        adapter.capture_images(["view"], 10)
