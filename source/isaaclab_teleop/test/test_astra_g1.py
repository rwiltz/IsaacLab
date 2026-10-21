# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect the G1 action and sensor-copy contracts without launching simulation."""

import base64
import copy
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.contrib.locomanip_pick_place.configs.pink_controller_cfg import G1_UPPER_BODY_IK_ACTION_CFG

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.g1 import G1ActionAdapter  # noqa: E402
from astra_teleop.perception import ObjectPerception, configure_object_contact_sensor  # noqa: E402
from teleop_astra_agent import run as run_astra  # noqa: E402


class Scene(dict):
    """Provide the scene's named-asset and origin interfaces over static CPU data."""


@pytest.fixture
def environment():
    hand_names = [
        f"{side}_hand_{joint}_joint"
        for side in ("left", "right")
        for joint in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
    ]
    joint_names = ["waist_yaw_joint", *reversed(hand_names), "left_knee_joint"]
    root_half = math.sqrt(0.5)
    data = SimpleNamespace(
        body_names=["right_wrist_yaw_link", "pelvis", "left_wrist_yaw_link"],
        body_link_pose_w=SimpleNamespace(
            torch=torch.tensor(
                [
                    [
                        [10.4, 19.8, 1.2, 0, 0, root_half, root_half],
                        [10, 20, 0.72, 0, 0, 0, 1],
                        [10.2, 20.3, 1.1, root_half, 0, 0, root_half],
                    ]
                ]
            )
        ),
        joint_names=joint_names,
        joint_pos=SimpleNamespace(torch=torch.zeros(1, 16)),
        joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-2.0, 2.0]] * 16])),
        soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-2.0, 2.0]] * 16])),
    )
    local_links = {
        "thumb_0": [0.06, -0.01, 0],
        "thumb_1": [0.06, -0.03, 0],
        "thumb_2": [0.06, -0.07, 0],
        "index_0": [0.12, 0, 0.03],
        "index_1": [0.16, 0, 0.03],
        "middle_0": [0.12, 0, -0.03],
        "middle_1": [0.16, 0, -0.03],
    }
    finger_poses = []
    for side, wrist_id in (("left", 2), ("right", 0)):
        for suffix, (x, y, z) in local_links.items():
            data.body_names.append(f"{side}_hand_{suffix}_link")
            pose = data.body_link_pose_w.torch[0, wrist_id].clone()
            # Known quarter turns: left +90 X, right +90 Z with mirrored hand Y.
            pose[:3] += torch.tensor([x, -z, y] if side == "left" else [y, x, z])
            finger_poses.append(pose)
    data.body_link_pose_w.torch = torch.cat(
        (data.body_link_pose_w.torch, torch.stack(finger_poses).unsqueeze(0)), dim=1
    )
    robot = SimpleNamespace(
        data=data, is_fixed_base=True, find_joints=lambda keys: resolve_matching_names(keys, joint_names)
    )
    terms = {
        "upper_body_ik": SimpleNamespace(cfg=G1_UPPER_BODY_IK_ACTION_CFG.copy()),
    }
    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
    scene.sensors = {
        "robot_pov_cam": SimpleNamespace(
            data=SimpleNamespace(
                output={"rgb": SimpleNamespace(torch=torch.zeros(1, 20, 30, 3, dtype=torch.uint8))},
                pos_w=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 1.5]])),
                quat_w_ros=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, root_half, root_half]])),
                intrinsic_matrices=SimpleNamespace(torch=torch.tensor([[[20.0, 0, 15], [0, 20, 10], [0, 0, 1]]])),
                image_shape=(20, 30),
            )
        ),
        "left_hand_contact": SimpleNamespace(
            body_names=["left_hand_index_1_link"],
            data=SimpleNamespace(net_forces_w=SimpleNamespace(torch=torch.tensor([[[1.0, 2.0, 3.0]]]))),
        ),
    }
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=scene,
        action_manager=SimpleNamespace(
            active_terms=["upper_body_ik"], action_term_dim=[28], get_term=terms.__getitem__
        ),
    )


def test_action_uses_environment_frame_xyzw_and_configured_task_order(environment):
    adapter = G1ActionAdapter(environment)
    idle = adapter.idle_action()
    assert idle.shape == (1, 28)
    assert idle[0, :3].tolist() == pytest.approx([0.2, 0.3, 1.1], abs=1e-6)
    assert idle[0, 7:10].tolist() == pytest.approx([0.4, -0.2, 1.2], abs=1e-6)
    assert idle[0, 3:7].tolist() == pytest.approx([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert idle[0, 10:14].tolist() == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])

    command = adapter.snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    action = adapter.action(command)
    assert action.shape == (1, 28)
    assert torch.equal(action, idle)
    tasks = environment.action_manager.get_term("upper_body_ik").cfg.controller.variable_input_tasks
    tasks[0], tasks[1] = tasks[1], tasks[0]
    with pytest.raises(ValueError, match="left then right"):
        G1ActionAdapter(environment)
    tasks[0], tasks[1] = tasks[1], tasks[0]
    environment.scene["robot"].is_fixed_base = False
    with pytest.raises(ValueError, match="requires a fixed-base robot"):
        G1ActionAdapter(environment)


def test_command_limits_and_invalid_quaternions(environment):
    environment.scene["robot"].data.body_link_pose_w.torch[0, 2, 3:] = torch.tensor([0, 0, 0, 1])
    adapter = G1ActionAdapter(environment)
    command = adapter.snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    origin = command["left_wrist"]["position"].copy()
    command["left_wrist"]["position"] = [origin[0] + 3, origin[1] + 4, origin[2]]
    command["left_wrist"]["quaternion"] = [0, 0, 1, 1]
    action = adapter.action(command)
    assert action[0, :3].tolist() == pytest.approx([origin[0] + 0.03, origin[1] + 0.04, origin[2]])
    assert action[0, 3:7].tolist() == pytest.approx([0, 0, math.sin(0.1), math.cos(0.1)])
    command["left_wrist"]["quaternion"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="nonzero norm"):
        adapter.action(command)
    command["left_wrist"]["quaternion"] = [0, 0, 0, float("nan")]
    with pytest.raises(ValueError, match="finite"):
        adapter.action(command)
    command["left_wrist"]["quaternion"] = [1e38] * 4
    with pytest.raises(ValueError, match="finite"):
        adapter.action(command)
    command["left_wrist"]["quaternion"] = [0, 0, 0, 1]
    command["left_grip_closed"] = 1
    with pytest.raises(ValueError, match="booleans"):
        adapter.action(command)
    for invalid_speed in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="Grip speed must be positive and finite"):
            G1ActionAdapter(environment, grip_speed=invalid_speed)


def test_grip_trajectory_joint_order_limits_reversal_and_idle(environment):
    robot = environment.scene["robot"]
    # Real G1 zero-angle startup lies outside soft limits on one-sided joints.
    # Restrict one finger more than the standard curl to exercise endpoint clipping.
    robot.data.joint_pos_limits.torch[0, 8] = torch.tensor([-0.45, 0])
    robot.data.soft_joint_pos_limits.torch[0, 8] = torch.tensor([-0.4, -0.08])
    robot.data.joint_pos_limits.torch[0, 4] = torch.tensor([0, 1])
    robot.data.soft_joint_pos_limits.torch[0, 4] = torch.tensor([0.05, 0.9])
    adapter = G1ActionAdapter(environment, grip_speed=0.8)
    command = adapter.snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    assert command["left_grip_closed"] is False and command["right_grip_closed"] is False
    assert adapter.grippers_open() == [True, True]
    command["left_grip_closed"] = True
    initial = adapter.action(command)
    # Measured fingers have not moved yet, but a closing command is not a released hand.
    assert adapter.grippers_open() == [False, True]
    action = initial
    expected = torch.tensor([0, 0, 0, 0.05, 0, 0, 0, -0.4, -0.9, -0.9, -0.9, 0.63, 0.36, 0])
    for step in range(60):
        previous = action
        action = adapter.hold_action(0.02) if step == 20 else adapter.advance_grip(action, 0.02)
        assert torch.all((action[:, 14:] - previous[:, 14:]).abs() <= 0.016001)
        assert torch.all((expected - action[0, 14:]).abs() <= (expected - previous[0, 14:]).abs() + 1e-6)
        if step == 10:
            # New model output must not restart from delayed measured positions.
            assert torch.equal(adapter.action(command)[:, 14:], action[:, 14:])
    assert action[0, 14:].tolist() == pytest.approx(expected.tolist(), abs=1e-6)
    assert torch.equal(action[:, :14], initial[:, :14])
    assert initial[:, 14:].count_nonzero() == 0
    # Contact can leave measured joints behind their targets. Waiting for inference
    # must retain the squeeze, while the wrists hold their current measured pose.
    robot.data.body_link_pose_w.torch[0, 2, 0] += 0.02
    held = adapter.hold_action(0.02)
    assert held[0, :3].tolist() == pytest.approx((initial[0, :3] + torch.tensor([0.02, 0, 0])).tolist(), abs=1e-6)
    assert held[0, 14:].tolist() == pytest.approx(expected.tolist(), abs=1e-6)

    command["left_grip_closed"] = False
    command["right_grip_closed"] = True
    reversed_action = adapter.action(command)
    assert torch.equal(reversed_action[:, 14:], held[:, 14:])
    for _ in range(60):
        previous = reversed_action
        reversed_action = adapter.advance_grip(reversed_action, 0.02)
        assert torch.all((reversed_action[:, 14:] - previous[:, 14:]).abs() <= 0.016001)
    assert reversed_action[0, 14:].tolist() == pytest.approx(
        [0.9, 0.9, 0.9, 0.9, -0.63, -0.36, 0, -0.08, 0, 0, 0, 0, 0, 0], abs=1e-6
    )
    idle = adapter.idle_action()
    assert idle[:, 14:].count_nonzero() == 0
    assert adapter.hold_action(0.02)[:, 14:].count_nonzero() == 0
    snapshot = adapter.snapshot({"policy": {}}, 0, 100, 2)
    assert snapshot["grip_trajectory_active"] is False
    assert snapshot["neutral_command"]["right_grip_closed"] is True
    adapter.action(command)
    adapter.reset()
    snapshot = adapter.snapshot({"policy": {}}, 1, 101, 2.02)
    assert snapshot["grip_trajectory_active"] is False
    assert snapshot["neutral_command"]["left_grip_closed"] is False
    assert snapshot["neutral_command"]["right_grip_closed"] is False
    # Opening is measured relative to the actual soft-clipped open endpoints,
    # which need not be zero, with each hand assessed independently.
    robot.data.joint_pos.torch[0, 8] = -0.17
    robot.data.joint_pos.torch[0, 4] = 0.14
    assert adapter.grippers_open() == [True, True]
    robot.data.joint_pos.torch[0, 8] = -0.19
    assert adapter.grippers_open() == [False, True]
    robot.data.joint_pos.torch[0, 8] = -0.08
    robot.data.joint_pos.torch[0, 4] = 0.16
    assert adapter.grippers_open() == [True, False]


@pytest.mark.parametrize(("grip_speed", "dwell_endpoints"), [(None, [0.8, 0.9]), (0.8, [0.4, 0.8, 0.9])])
def test_configured_grip_speed_completes_closure_across_half_second_waypoints(environment, grip_speed, dwell_endpoints):
    adapter = (
        G1ActionAdapter(environment) if grip_speed is None else G1ActionAdapter(environment, grip_speed=grip_speed)
    )
    assert adapter.describe()["limits"]["hand_joint_target_speed_rad_s"] == (1.6 if grip_speed is None else 0.8)
    command = adapter.snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    command["left_grip_closed"] = True
    command["right_grip_closed"] = True
    endpoints = adapter.describe()["grip_targets_in_action_order_rad"]["closed"]
    for index, expected_curl in enumerate(dwell_endpoints):
        # New waypoints retain the target ramp even when measured joints lag behind it.
        action = adapter.action(command)
        for _ in range(25):
            action = adapter.advance_grip(action, 0.02)
        assert float(action[0, 14:].abs().max()) == pytest.approx(expected_curl, abs=1e-6)
        if index < len(dwell_endpoints) - 1:
            assert action[0, 14:].tolist() != pytest.approx(endpoints)
    assert action[0, 14:].tolist() == pytest.approx(endpoints, abs=1e-6)


@pytest.mark.parametrize(("chunk_size", "wrist_speed"), [(1, 10.0), (3, 10.0), (3, 0.5)])
def test_runner_wait_preemption_and_expiry_preserve_correct_targets(environment, chunk_size, wrist_speed):
    command = G1ActionAdapter(environment).snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    initial_x = command["left_wrist"]["position"][0]
    command["left_wrist"]["position"][0] += 0.03
    command["left_grip_closed"] = True
    sent = []
    resets = []
    event_queue = []

    class ScriptedPipeline:
        """Supply inference transitions while the production runner and adapter choose actions."""

        config = SimpleNamespace(action_chunk_size=chunk_size, waypoint_seconds=1.0, model="mock", goal="Reach")

        def needs_snapshot(self):
            return False

        def poll(self, **kwargs):
            return command if len(sent) in (0, 6) else None

        @property
        def status(self):
            return {
                "plan": (
                    {"subgoal": "Approach" if len(sent) < 2 else "Inspect", "stop": 4 <= len(sent) < 6}
                    if len(sent) < 7
                    else None
                ),
                "command_id": 1 if len(sent) < 6 else 2,
                "chunk_id": None,
                "waypoint_index": None,
            }

        def drain_events(self):
            events = event_queue.copy()
            event_queue.clear()
            return events

        def advance(self, dt):
            if len(sent) == 7 and chunk_size > 1:
                event_queue.append({"event": "feedback_checkpoint", "chunk_id": 3, "completed_waypoints": 1})

        def reset(self):
            resets.append(len(sent))

    def step(action):
        sent.append(action.clone())
        # Measured fingers lag their targets; wrist drift distinguishes a fixed
        # accepted hold from following measurements on each inference-wait tick.
        data = environment.scene["robot"].data
        data.body_link_pose_w.torch[0, 2, 0] += 0.01
        data.joint_pos.torch += 0.001
        terminated = len(sent) == 7
        if terminated:
            # ManagerBasedRLEnv has already reset its measured state when step returns.
            data.body_link_pose_w.torch[0, 2, 0] = environment.scene.env_origins[0, 0] + initial_x + 0.5
            data.joint_pos.torch.fill_(0.2)
        return {}, None, torch.tensor([terminated]), torch.tensor([False]), {}

    environment.reset = lambda: ({}, {})
    environment.step = step
    environment.step_dt = 0.02
    environment.sim = SimpleNamespace(is_running=lambda: True)
    args = SimpleNamespace(
        mock=True,
        bin_drop=False,
        reset_on_success=False,
        max_episodes=0,
        pause_during_inference=False,
        cameras=[],
        trace=None,
        image_trace_dir=None,
        max_steps=9,
        status_interval=0,
        wrist_linear_speed=wrist_speed,
        wrist_angular_speed=10.0,
        grip_speed=0.8,
    )
    trace = io.StringIO()
    run_astra(environment, ScriptedPipeline(), args, trace)

    # A checkpoint emitted by the terminating physics step belongs to that
    # episode, even though ordinary event collection happens before env.step.
    outcomes = [
        event
        for line in trace.getvalue().splitlines()
        for event in json.loads(line)["events"]
        if event["event"] == "episode_completed"
    ]
    assert len(outcomes) == 1
    assert outcomes[0]["episode_id"] == 0
    assert outcomes[0]["feedback_checkpoints"] == int(chunk_size > 1)

    # Slow execution reaches only 1 cm of the 3 cm target before its allocated
    # dwell ends. Inference waits must freeze there, not finish that discarded path.
    if chunk_size > 1:
        reached = 0.01 if wrist_speed == 0.5 else 0.03
        expected_offsets = [reached, reached, 0.02, 0.02, 0.04, 0.04, 0.03, 0.5, 0.5]
    else:
        expected_offsets = [0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.5, 0.5]
    assert [float(action[0, 0]) - initial_x for action in sent] == pytest.approx(expected_offsets, abs=2e-6)
    squeeze = [float(action[0, 14:].abs().sum()) for action in sent]
    assert squeeze[0] > 0
    if chunk_size > 1:
        assert all(after > before for before, after in zip(squeeze[:3], squeeze[1:4], strict=True))
    # An idle hold captures measured fingers once and cancels the old squeeze goal.
    idle_start = 4 if chunk_size > 1 else 1
    for action in sent[idle_start:6]:
        assert action[0, 14:].tolist() == pytest.approx([idle_start * 0.001] * 14)
    assert sent[6][0, 14:].tolist() != pytest.approx([0.006] * 14)  # A new action restarts the grip trajectory.
    assert resets == [7]
    for action in sent[7:]:
        assert action[0, 14:].tolist() == pytest.approx([0.2] * 14)


def test_runner_pauses_physics_until_reply_but_executes_during_prefetch(environment, monkeypatch, tmp_path):
    command = G1ActionAdapter(environment).snapshot({"policy": {}}, 0, 0, 0)["neutral_command"]
    initial_x = command["left_wrist"]["position"][0]
    command["left_wrist"]["position"][0] += 0.04
    command["left_grip_closed"] = True
    sent, advanced, rendered_at_steps, execution_modes = [], [], [], []

    class DelayedPipeline:
        """Make replies ready after UI updates, independently of physics stepping."""

        config = SimpleNamespace(action_chunk_size=3, waypoint_seconds=1.0, model="mock", goal="Reach")
        polls = 0

        def needs_snapshot(self):
            return self.polls < 2

        def poll(self, **kwargs):
            self.polls += 1
            execution_modes.append(kwargs["action_spec"]["execution_timing"]["pause_during_inference"])
            ready = len(rendered_at_steps) >= (3 if len(sent) < 2 else 6)
            return command if ready else None

        @property
        def status(self):
            return {
                "plan": {"subgoal": "Reach", "stop": False},
                "command_id": 1 if len(sent) < 2 else 2,
                "chunk_id": 1 if len(sent) < 2 else 2,
                "waypoint_index": 0,
                "in_flight": ["system1"],  # Successor prefetch must not pause a usable current command.
            }

        def drain_events(self):
            return []

        def advance(self, dt):
            advanced.append(dt)

    def step(action):
        sent.append(action.clone())
        return {}, None, torch.tensor([False]), torch.tensor([False]), {}

    def render():
        rendered_at_steps.append(len(sent))
        environment.scene.sensors["robot_pov_cam"].data.output["rgb"].torch.fill_(len(rendered_at_steps) * 30)

    pipeline = DelayedPipeline()
    environment.reset = lambda: ({"policy": {}}, {})
    environment.step = step
    environment.step_dt = 0.02
    environment.sim = SimpleNamespace(
        is_running=lambda: pipeline.polls < 20,
        render=render,
    )
    args = SimpleNamespace(
        mock=True,
        bin_drop=False,
        reset_on_success=False,
        max_episodes=0,
        pause_during_inference=True,
        cameras=["robot_pov_cam"],
        image_size=15,
        trace=None,
        image_trace_dir=tmp_path,
        max_steps=4,
        status_interval=0,
        wrist_linear_speed=0.5,
        wrist_angular_speed=10.0,
        grip_speed=0.8,
    )
    monkeypatch.setattr("teleop_astra_agent.time.sleep", lambda _seconds: None)
    trace = io.StringIO()
    run_astra(environment, pipeline, args, trace)

    assert execution_modes and all(execution_modes)
    assert rendered_at_steps == [0, 0, 0, 2, 2, 2]
    assert advanced == [0.02] * 4
    assert len(sent) == 4
    # Neither wrist interpolation nor the closing trajectory may run ahead while physics is frozen.
    assert [float(action[0, 0]) - initial_x for action in sent] == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert [float(action[0, 14:].abs().max()) for action in sent] == pytest.approx([0.016, 0.032, 0.048, 0.064])
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [event["event"] for row in rows for event in row["events"]] == [
        "inference_paused",
        "inference_resumed",
        "inference_paused",
        "inference_resumed",
    ]
    assert [row["step"] for row in rows if row["inference_paused"]] == [0, 0, 2]
    assert all(row["action"] is None for row in rows if row["inference_paused"])

    # Separate S2/S1 captures at the same frozen physics step must remain recoverable.
    captures = sorted(tmp_path.glob("*.jpg"))
    assert len(captures) == 2
    assert captures[0].read_bytes() != captures[1].read_bytes()
    assert [row["capture_id"] for row in rows if row["snapshot"] is not None] == [1, 2]


def test_verified_success_resets_once_and_starts_fresh_episode_metrics(environment, monkeypatch):
    clock = [100.0]
    sent, reset_steps, observer_resets, observer_updates = [], [], [], []
    step_in_episode = 0
    event_queue = []
    monkeypatch.setattr("teleop_astra_agent.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("teleop_astra_agent.time.sleep", lambda _seconds: None)

    class PlacementObserver:
        """Supply completion decisions at the simulator observation boundary."""

        task = SimpleNamespace(describe=lambda: {"task": "Drop into the authored bin"})

        def update(self, dt):
            observer_updates.append((len(reset_steps), dt))
            return {"success": step_in_episode == 2, "settled_sim_seconds": step_in_episode * 0.5}

        def reset(self):
            observer_resets.append(len(reset_steps))

    monkeypatch.setattr("astra_teleop.perception.BinDropObserver", lambda *_args: PlacementObserver())

    class EpisodePipeline:
        config = SimpleNamespace(action_chunk_size=2, waypoint_seconds=1.0, model="mock", goal="Drop")
        episode = 0
        resets = 0

        def needs_snapshot(self):
            return True

        def poll(self, snapshot, **kwargs):
            if step_in_episode == 0:
                latency = 0.25 * (self.episode + 1)
                event_queue.append({"event": "submitted", "role": "system1"})
                clock[0] += latency
                event_queue.append(
                    {"event": "accepted", "role": "system1", "latency_seconds": latency, "result": {"stop": False}}
                )
            return snapshot["neutral_command"]

        @property
        def status(self):
            return {
                "plan": {"subgoal": "Release and settle", "stop": False},
                "command_id": self.episode + 1,
                "chunk_id": self.episode + 1,
                "waypoint_index": 0,
            }

        def advance(self, dt):
            if step_in_episode == 2:
                event_queue.append(
                    {"event": "feedback_checkpoint", "chunk_id": self.episode + 1, "completed_waypoints": 2}
                )

        def drain_events(self):
            events = event_queue.copy()
            event_queue.clear()
            return events

        def reset(self):
            self.episode += 1
            self.resets += 1

    def reset():
        nonlocal step_in_episode
        reset_steps.append(len(sent))
        step_in_episode = 0
        clock[0] += 5.0  # Reset/setup time must not inflate the following attempt's ready-time metrics.
        environment.scene["robot"].data.body_link_pose_w.torch[0, 2, 0] = 10.2 + 0.5 * (len(reset_steps) - 1)
        return {"policy": {}}, {}

    def step(action):
        nonlocal step_in_episode
        sent.append(action.clone())
        step_in_episode += 1
        clock[0] += 0.1
        return {"policy": {}}, None, torch.tensor([False]), torch.tensor([False]), {}

    environment.reset = reset
    environment.step = step
    environment.step_dt = 0.5
    environment.sim = SimpleNamespace(is_running=lambda: True)
    args = SimpleNamespace(
        mock=True,
        bin_drop=True,
        original_success_cfg=None,
        reset_on_success=True,
        max_episodes=2,
        pause_during_inference=False,
        cameras=[],
        image_size=15,
        trace=None,
        image_trace_dir=None,
        max_steps=8,
        status_interval=0,
        wrist_linear_speed=0.025,
        wrist_angular_speed=0.4,
        grip_speed=0.8,
    )
    pipeline = EpisodePipeline()
    trace = io.StringIO()
    run_astra(environment, pipeline, args, trace)

    assert reset_steps == [0, 2]  # Initial reset, then one explicit success reset; none after the episode limit.
    assert pipeline.resets == 1
    assert observer_resets == [2]
    assert observer_updates == [(1, 0), (1, 0.5), (1, 0.5), (2, 0), (2, 0.5), (2, 0.5)]
    assert [action[0, 0].item() for action in sent] == pytest.approx([0.2, 0.2, 0.7, 0.7], abs=1e-6)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    outcomes = [event for row in rows for event in row["events"] if event["event"] == "episode_completed"]
    assert [outcome["episode_id"] for outcome in outcomes] == [0, 1]
    assert all(outcome["outcome"] == "verified_bin_drop" for outcome in outcomes)
    assert [outcome["feedback_checkpoints"] for outcome in outcomes] == [1, 1]
    assert [outcome["simulation_seconds"] for outcome in outcomes] == [1.0, 1.0]
    assert [outcome["wall_seconds"] for outcome in outcomes] == pytest.approx([0.45, 0.7])
    assert [outcome["ready_to_first_action_seconds"] for outcome in outcomes] == pytest.approx([0.25, 0.5])
    assert [outcome["accepted_request_latency_seconds"]["system1"] for outcome in outcomes] == [[0.25], [0.5]]
    assert [outcome["requests_submitted"]["system1"] for outcome in outcomes] == [1, 1]
    first_next_snapshot = next(row["snapshot"] for row in rows if row["episode_id"] == 1 and row.get("snapshot"))
    assert first_next_snapshot["task_progress"]["success"] is False
    assert first_next_snapshot["task_progress"]["settled_sim_seconds"] == 0


def test_sensor_snapshots_are_detached_and_camera_is_resized(environment):
    adapter = G1ActionAdapter(environment)
    pixels = environment.scene.sensors["robot_pov_cam"].data.output["rgb"].torch
    state = torch.tensor([[0.3, 0.5, 1.4]])
    snapshot = adapter.snapshot({"policy": {"object_pos": state, "robot_pov_cam": pixels}}, 3, 4, 0.08)
    images = adapter.capture_images(["robot_pov_cam"], max_size=15)
    state.fill_(99)
    pixels.fill_(255)
    environment.scene["robot"].data.body_link_pose_w.torch.zero_()
    environment.scene.sensors["robot_pov_cam"].data.pos_w.torch.zero_()
    environment.scene.sensors["left_hand_contact"].data.net_forces_w.torch.zero_()

    assert snapshot["observations"]["object_pos"] == pytest.approx([0.3, 0.5, 1.4])
    assert snapshot["neutral_command"]["left_wrist"]["position"] == pytest.approx([0.2, 0.3, 1.1], abs=1e-6)
    left = snapshot["hand_geometry"]["left"]
    assert left["wrist_axes_environment"]["y"] == pytest.approx([0, 0, 1])
    assert left["wrist_axes_environment"]["z"] == pytest.approx([0, -1, 0])
    assert left["links"]["index_1"]["position_environment_m"] == pytest.approx([0.36, 0.27, 1.1], abs=2e-6)
    assert left["links"]["thumb_2"]["position_wrist_m"] == pytest.approx([0.06, -0.07, 0], abs=1e-6)
    assert left["object_origin_wrist_m"] == pytest.approx([0.1, 0.3, -0.2], abs=1e-6)
    right = snapshot["hand_geometry"]["right"]
    assert right["wrist_axes_environment"]["x"] == pytest.approx([0, 1, 0])
    assert right["links"]["thumb_2"]["position_wrist_m"] == pytest.approx([0.06, 0.07, 0], abs=1e-6)
    assert snapshot["hand_contact_body_names"]["left_hand_contact"] == ["left_hand_index_1_link"]
    assert snapshot["hand_contact_forces_world_n"]["left_hand_contact"] == [[1.0, 2.0, 3.0]]
    camera = snapshot["cameras"]["robot_pov_cam"]
    assert camera["position_environment_m"] == pytest.approx([0, 0, 1.5])
    assert camera["quaternion_ros_xyzw"] == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    assert camera["image_size_native_wh"] == [30, 20]
    assert camera["intrinsic_matrix_native_pixels"] == [[20, 0, 15], [0, 20, 10], [0, 0, 1]]
    assert (snapshot["episode_id"], snapshot["step"], snapshot["sim_time"]) == (3, 4, 0.08)
    json.dumps(snapshot, allow_nan=False)
    assert images[0]["name"] == "robot_pov_cam"
    with Image.open(io.BytesIO(base64.b64decode(images[0]["data_url"].split(",", 1)[1]))) as frame:
        assert frame.size == (15, 10)
        assert frame.getpixel((0, 0)) == (0, 0, 0)


def test_object_motion_tracks_lift_relative_slip_and_resets_baselines(environment):
    pose = torch.tensor([[10.3, 20.4, 0.7, 0, 0, 0, 1]])
    environment.scene["object"] = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose)))
    wrists = environment.scene["robot"].data.body_link_pose_w.torch
    contacts = {
        "left_hand_index_1_link": [0.3, 0.27, 1.1],
        "right_hand_middle_1_link": [0.4, -0.1, 1.2],
        "left_hand_thumb_2_link": None,
    }
    perception = SimpleNamespace(snapshot=lambda: {"object_contact_positions_environment_m": copy.deepcopy(contacts)})
    adapter = G1ActionAdapter(environment, perception=perception)
    # The baseline must precede even the first inference snapshot.
    pose[0, 2] += 0.02
    first = adapter.snapshot({"policy": {}}, 0, 10, 0.2)["object_motion"]
    assert first["episode_initial_pose"]["position_environment_m"] == pytest.approx([0.3, 0.4, 0.7], abs=1e-6)
    assert first["height_change_from_episode_start_m"] == pytest.approx(0.02)
    assert first["since_previous_snapshot"] is None
    assert "linear_velocity_world_m_s" not in first
    first["episode_initial_pose"]["position_environment_m"][2] = 99  # Detached from the retained baseline.

    pose[0, 2] += 0.05
    wrists[0, [0, 2], 2] += 0.05
    for position in contacts.values():
        if position is not None:
            position[2] += 0.05
    contacts["left_hand_thumb_2_link"] = [0.3, 0.25, 1.15]  # Newly reported contact has no preceding centroid.
    velocity = torch.tensor([[0.01, -0.02, 0.03, -0.1, 0.2, 0.3]])
    environment.scene["object"].data.root_link_vel_w = SimpleNamespace(torch=velocity)
    lifted = adapter.snapshot({"policy": {}}, 0, 110, 2.2)["object_motion"]
    velocity.fill_(0)
    delta = lifted["since_previous_snapshot"]
    assert lifted["height_change_from_episode_start_m"] == pytest.approx(0.07)
    assert delta["elapsed_sim_seconds"] == pytest.approx(2.0)
    assert delta["object_translation_environment_m"] == pytest.approx([0, 0, 0.05])
    assert delta["object_rotation_rad"] == pytest.approx(0)
    assert lifted["linear_velocity_world_m_s"] == pytest.approx([0.01, -0.02, 0.03])
    assert lifted["angular_velocity_world_rad_s"] == pytest.approx([-0.1, 0.2, 0.3])
    contact_delta = delta["contact_centroid_translation_wrist_m"]
    assert set(contact_delta) == {"left_hand_index_1_link", "right_hand_middle_1_link"}
    for displacement in contact_delta.values():
        assert displacement == pytest.approx([0, 0, 0], abs=1e-6)
    for hand in delta["wrists"].values():
        assert hand["translation_environment_m"] == pytest.approx([0, 0, 0.05])
        assert hand["object_relative_translation_wrist_m"] == pytest.approx([0, 0, 0], abs=1e-6)
        assert hand["object_relative_rotation_rad"] == pytest.approx(0, abs=1e-6)

    # The left wrist has a known +90-degree X rotation: world -Z slip is local -Y.
    pose[0, 2] -= 0.01
    pose[0, 3:] = torch.tensor([math.sin(0.1), 0, 0, math.cos(0.1)])
    contacts["left_hand_index_1_link"][2] -= 0.005
    contacts["right_hand_middle_1_link"] = None
    contacts["left_hand_thumb_2_link"] = None
    slipped = adapter.snapshot({"policy": {}}, 0, 135, 2.7)["object_motion"]["since_previous_snapshot"]
    assert slipped["previous_sim_time_s"] == pytest.approx(2.2)
    assert slipped["elapsed_sim_seconds"] == pytest.approx(0.5)
    assert slipped["object_rotation_rad"] == pytest.approx(0.2, abs=1e-6)
    assert slipped["wrists"]["left"]["object_relative_translation_wrist_m"] == pytest.approx([0, -0.01, 0], abs=1e-6)
    assert slipped["wrists"]["left"]["object_relative_rotation_rad"] == pytest.approx(0.2, abs=1e-6)
    assert slipped["wrists"]["left"]["translation_environment_m"] == [0, 0, 0]
    assert set(slipped["contact_centroid_translation_wrist_m"]) == {"left_hand_index_1_link"}
    assert slipped["contact_centroid_translation_wrist_m"]["left_hand_index_1_link"] == pytest.approx(
        [0, -0.005, 0], abs=1e-6
    )

    pose[0, 2] = 0.9
    adapter.reset()
    pose[0, 2] += 0.03
    reset = adapter.snapshot({"policy": {}}, 1, 150, 3.0)["object_motion"]
    assert reset["episode_initial_pose"]["position_environment_m"][2] == pytest.approx(0.9)
    assert reset["height_change_from_episode_start_m"] == pytest.approx(0.03)
    assert reset["since_previous_snapshot"] is None
    json.dumps(reset, allow_nan=False)


def test_object_perception_preserves_scaled_geometry_and_labeled_contacts(environment, monkeypatch):
    from pxr import Usd, UsdGeom

    import isaaclab.sim.utils as sim_utils

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Object")
    root.AddTranslateOp().Set((10, 20, 0.7))
    root.AddRotateZOp().Set(90)
    root.AddScaleOp().Set((0.75, 0.75, 0.75))
    body = UsdGeom.Xform.Define(stage, "/World/Object/Geometry/body")
    body.AddTranslateOp().Set((0, 0, 0.1))
    rim = UsdGeom.Mesh.Define(stage, "/World/Object/Geometry/body/wheel_rim_01")
    UsdGeom.Xformable(rim).AddTranslateOp().Set((0, 0, 0.2))
    # The authored rim also contains center detail; its overall Z minimum is
    # below the outer band. Keep those distinct instead of treating the AABB as a torus.
    rim.CreatePointsAttr([(1, 0, 0.1), (-1, 0, 0.1), (0, 1, 0.3), (0, -1, 0.3), (0.02, 0, 0), (-0.02, 0, 0)])
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)
    scene_cfg = SimpleNamespace(
        object=SimpleNamespace(prim_path="/World/Object", spawn=SimpleNamespace(activate_contact_sensors=False)),
        robot=SimpleNamespace(prim_path="/World/Robot"),
    )
    configure_object_contact_sensor(scene_cfg)
    forces = torch.arange(42, dtype=torch.float32).reshape(1, 1, 14, 3)
    sensor = SimpleNamespace(
        cfg=scene_cfg.object_hand_contact,
        num_sensors=1,
        body_physx_view=SimpleNamespace(prim_paths=[str(body.GetPath())]),
        data=SimpleNamespace(
            force_matrix_w=SimpleNamespace(torch=forces),
            net_forces_w=SimpleNamespace(torch=forces.sum(dim=2)),
            contact_pos_w=SimpleNamespace(torch=torch.full((1, 1, 14, 3), float("nan"))),
        ),
    )
    environment.scene.sensors["object_hand_contact"] = sensor
    perception = ObjectPerception(environment)
    snapshot = G1ActionAdapter(environment, perception=perception).snapshot({"policy": {}}, 0, 0, 0)
    forces.fill_(99)
    perception.geometry["parts"].clear()

    assert scene_cfg.object.spawn.activate_contact_sensors
    assert snapshot["object_contact_forces_world_n"]["left_hand_middle_1_link"] == [18, 19, 20]
    assert snapshot["object_contact_forces_world_n"]["right_hand_thumb_0_link"] == [21, 22, 23]
    geometry = snapshot["object_geometry"]
    assert geometry["bounds_m"][0] == pytest.approx([-0.75, -0.75, 0.15])
    assert geometry["bounds_m"][1] == pytest.approx([0.75, 0.75, 0.375])
    band = geometry["parts"][0]["outer_radial_band"]
    assert band["radius_range_m"] == pytest.approx([0.75, 0.75])
    assert band["z_range_m"] == pytest.approx([0.225, 0.375])
    json.dumps(snapshot, allow_nan=False)
    sensor.data.force_matrix_w = SimpleNamespace(torch=torch.zeros(1, 1, 1, 3))
    with pytest.raises(ValueError, match="one force column"):
        perception.snapshot()
