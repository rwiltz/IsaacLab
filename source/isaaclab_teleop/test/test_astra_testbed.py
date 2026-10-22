# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check generic runner physics and episode boundaries without launching a simulator."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderManager

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
import run_astra_test  # noqa: E402
from astra_teleop.core import ActionContract  # noqa: E402


class EnvironmentBoundary:
    """Represent manager-environment autoreset behavior at the runner boundary."""

    step_dt = 0.02
    num_envs = 1
    device = "cpu"

    def __init__(self, native_termination=False):
        self.native_termination = native_termination
        self.actions = []
        self.explicit_resets = 0
        self.generation = 0
        self.render_step_counts = []
        self.sim = SimpleNamespace(is_running=lambda: True, is_playing=lambda: True, render=self.render)
        self.cfg = SimpleNamespace(
            env_name="Test-Task",
            sim=SimpleNamespace(dt=self.step_dt, render_interval=1),
            decimation=1,
            scene=SimpleNamespace(num_envs=1),
        )
        self.action_manager = SimpleNamespace(action=torch.zeros(1, 2))
        self.recorder_manager = RecorderManager(
            ActionStateRecorderManagerCfg(
                dataset_export_mode=DatasetExportMode.EXPORT_NONE,
                record_initial_state=None,
                record_post_step_states=None,
                record_pre_step_flat_policy_observations=None,
                record_post_step_processed_actions=None,
            ),
            self,
        )

    def render(self):
        self.render_step_counts.append(len(self.actions))

    def step(self, action):
        self.action_manager.action = action
        self.recorder_manager.record_pre_step()
        self.actions.append(action.clone())
        if self.native_termination:
            self.recorder_manager.record_pre_reset([0])
            self.recorder_manager.reset([0])
            self.generation += 1
        return (
            {"generation": self.generation},
            0.0,
            torch.tensor([self.native_termination]),
            torch.tensor([False]),
            {},
        )

    def reset(self):
        self.recorder_manager.record_pre_reset([0])
        self.recorder_manager.reset([0])
        self.explicit_resets += 1
        self.generation += 1
        return {"generation": self.generation}, {}


class AdapterBoundary:
    """Supply an arbitrary two-value command and independently measured completion."""

    def __init__(self, success_after_step=False):
        self.contract = ActionContract(
            command_schema={
                "type": "object",
                "properties": {
                    "thrust": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -1, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["thrust"],
                "additionalProperties": False,
            },
            instructions="Command two normalized thrust values.",
        )
        self.success_after_step = success_after_step
        self.progress_dts = []
        self.reset_count = 0

    def describe(self):
        return {"command": "Two normalized thrust values"}

    def snapshot(self, observations, episode_id, step, sim_time):
        return {"observations": observations, "episode_id": episode_id, "neutral_command": {"thrust": [0.0, 0.0]}}

    def capture_images(self, cameras, image_size):
        return []

    def action(self, command):
        return torch.tensor([command["thrust"]])

    def idle_action(self):
        return torch.zeros(1, 2)

    def task_progress(self, dt):
        self.progress_dts.append(dt)
        return {"success": self.success_after_step and dt > 0}

    def reset(self):
        self.reset_count += 1


class InferenceBoundary:
    """Provide a pending worker followed by a non-G1 structured command."""

    def __init__(self, pending_polls=0):
        self.pending_polls = pending_polls
        self.status = {"in_flight": []}
        self.snapshots = []
        self.advance_dts = []
        self.reset_count = 0

    def needs_snapshot(self):
        return True

    def poll(self, *, snapshot, images, action_spec):
        self.snapshots.append(snapshot)
        if self.pending_polls:
            self.pending_polls -= 1
            self.status = {"in_flight": ["system1"]}
            return None
        self.status = {"in_flight": []}
        return {"thrust": [0.25, -0.5]}

    def drain_events(self):
        return []

    def advance(self, dt):
        self.advance_dts.append(dt)

    def reset(self):
        self.reset_count += 1


@pytest.fixture
def runner_args(monkeypatch):
    monkeypatch.setattr(run_astra_test.time, "sleep", lambda duration: None)
    return SimpleNamespace(
        task="Test-Task",
        adapter="test:AdapterBoundary",
        mock=True,
        trace=Path("unused.jsonl"),
        image_trace_dir=None,
        dataset_file=None,
        cameras=[],
        image_size=640,
        action_chunk_size=8,
        pause_during_inference=True,
        status_interval=0.0,
        max_steps=4,
        max_episodes=2,
        reset_on_success=True,
    )


def test_pending_inference_renders_without_stepping_then_executes_non_g1_action(runner_args):
    env = EnvironmentBoundary()
    adapter = AdapterBoundary()
    pipeline = InferenceBoundary(pending_polls=2)
    runner_args.max_steps = 1
    trace = io.StringIO()

    run_astra_test.run(env, pipeline, adapter, {"generation": 0}, runner_args, trace)

    assert env.render_step_counts == [0, 0]
    assert len(env.actions) == 1
    assert env.actions[0].tolist() == [[0.25, -0.5]]
    assert torch.stack(env.recorder_manager.get_episode(0).data["actions"]).tolist() == [[0.25, -0.5]]
    assert pipeline.advance_dts == [env.step_dt]
    assert adapter.progress_dts == [0.0, env.step_dt]
    control_rows = [row for row in map(json.loads, trace.getvalue().splitlines()) if "action" in row]
    assert [row["inference_paused"] for row in control_rows] == [True, True, False]
    assert [row["action"] for row in control_rows] == [None, None, [[0.25, -0.5]]]
    assert all(row["step"] == 0 for row in control_rows)


@pytest.mark.parametrize("native_termination, reset_on_success", [(True, True), (False, True), (False, False)])
def test_native_autoreset_and_verified_success_reset_preserve_episode_boundaries(
    runner_args, monkeypatch, native_termination, reset_on_success
):
    env = EnvironmentBoundary(native_termination=native_termination)
    adapter = AdapterBoundary(success_after_step=True)
    pipeline = InferenceBoundary()
    trace = io.StringIO()
    runner_args.dataset_file = Path("unused.hdf5")
    runner_args.reset_on_success = reset_on_success
    exports = []
    export_episodes = env.recorder_manager.export_episodes

    def capture_export(*args, **kwargs):
        episode = env.recorder_manager.get_episode(0)
        if len(episode.data.get("actions", [])):
            exports.append((env.generation, episode))
        return export_episodes(*args, **kwargs)

    monkeypatch.setattr(env.recorder_manager, "export_episodes", capture_export)

    run_astra_test.run(env, pipeline, adapter, {"generation": 0}, runner_args, trace)

    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    episodes = [row for row in rows if row.get("event") == "episode_completed"]
    expected_episodes = 2 if reset_on_success else 1
    expected_ids = list(range(expected_episodes))
    assert len(env.actions) == expected_episodes
    assert [row["episode_id"] for row in episodes] == expected_ids
    assert [snapshot["observations"]["generation"] for snapshot in pipeline.snapshots] == expected_ids
    assert [snapshot["episode_id"] for snapshot in pipeline.snapshots] == expected_ids
    assert pipeline.reset_count == adapter.reset_count == expected_episodes - 1
    assert env.explicit_resets == (0 if native_termination else expected_episodes - 1)
    assert [generation for generation, _ in exports] == expected_ids
    for _, recorded in exports:
        assert recorded.success is (not native_termination)
        assert recorded.data["actions"].tolist() == [[0.25, -0.5]]
    assert all(row["simulation_seconds"] == env.step_dt for row in episodes)
    if native_termination:
        assert all(row["outcome"] == "environment_termination" and row["task_progress"] is None for row in episodes)
        assert adapter.progress_dts == [0.0, 0.0]
    else:
        assert all(row["outcome"] == "verified_success" and row["task_progress"]["success"] for row in episodes)
        assert adapter.progress_dts == [0.0, env.step_dt] * expected_episodes
