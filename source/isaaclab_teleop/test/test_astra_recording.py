# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify Astra episode exports using the real recorder and HDF5 writer, without a simulator."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderManager, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.recording import (  # noqa: E402
    close_recording,
    configure_recording,
    export_recording,
    reset_after_export,
)


class ResetSummaryRecorder(RecorderTerm):
    """Record a task-specific terminal measurement before each reset."""

    def record_pre_reset(self, env_ids):
        return "terminal_position", self._env.pose[env_ids, :3]


@configclass
class StateOnlyRecorderCfg(ActionStateRecorderManagerCfg):
    """Represent a task with reset summaries and no policy observations in its dataset."""

    record_pre_step_flat_policy_observations = None
    record_reset_summary = RecorderTermCfg(class_type=ResetSummaryRecorder)


class RecordedEnvironment:
    """Supply prescribed control and state samples to the standard recorder terms."""

    num_envs = 1
    device = "cpu"

    def __init__(self, cfg):
        self.cfg = cfg
        self.sim = SimpleNamespace(is_playing=lambda: True)
        self.pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        self.scene = SimpleNamespace(get_state=self.get_state)
        self.native_success = torch.tensor([False])
        self.termination_manager = SimpleNamespace(active_terms=["success"], get_term=lambda name: self.native_success)
        self.action_manager = SimpleNamespace(
            active_terms=["control"],
            action=torch.zeros(1, 2),
            processed_actions=torch.zeros(1, 2),
        )
        self.action_manager.get_term = lambda name: self.action_manager
        self.obs_buf = {"policy": torch.tensor([[4.0, 5.0, 6.0]])}
        self.recorder_manager = RecorderManager(cfg.recorders, self)
        self.recorder_manager.record_post_reset([0])

    def get_state(self, *, is_relative):
        assert is_relative
        return {"rigid_object": {"object": {"root_pose": self.pose}}}

    def record_step(self, action, position_x):
        self.action_manager.action = torch.tensor([action])
        self.action_manager.processed_actions = torch.tensor([[7.0, 8.0]])
        self.recorder_manager.record_pre_step()
        self.pose[0, 0] = position_x
        self.recorder_manager.record_post_step()

    def reset(self):
        self.recorder_manager.record_pre_reset([0])
        self.recorder_manager.reset([0])
        self.native_success.fill_(False)
        self.pose[0, 0] = 1.0
        self.recorder_manager.record_post_reset([0])
        return self.obs_buf, {}


@pytest.fixture
def recording_env(tmp_path, monkeypatch):
    managers = []

    def create(export_mode="success", recorder_cfg=None, filename="demonstrations"):
        task = "Astra-Recording-Test-v0"
        kwargs = {} if recorder_cfg is None else {"demo_recorder_cfg_entry_point": recorder_cfg}
        monkeypatch.setitem(gym.registry, task, gym.envs.registration.EnvSpec(id=task, kwargs=kwargs))
        cfg = SimpleNamespace(
            env_name=task,
            sim=SimpleNamespace(dt=0.01, render_interval=2),
            decimation=2,
            scene=SimpleNamespace(num_envs=1),
            observations=SimpleNamespace(policy=SimpleNamespace(concatenate_terms=True)),
            terminations=SimpleNamespace(time_out=object(), success=object()),
            commands=SimpleNamespace(resampling_time_range=(1.0, 2.0)),
        )
        path = configure_recording(cfg, task, tmp_path / filename, export_mode)
        env = RecordedEnvironment(cfg)
        managers.append(env.recorder_manager)
        return env, path

    yield create
    for manager in managers:
        manager.close()


@pytest.mark.parametrize("export_mode", ["success", "all"])
def test_saved_episodes_preserve_actions_states_and_filter_interrupted_attempts(recording_env, export_mode):
    env, path = recording_env(export_mode)
    assert path.name == "demonstrations.hdf5"

    # Initial-state-only buffers, duplicate finalization, and the next reset must not add empty demos.
    export_recording(env)
    env.record_step([0.25, -0.5], 2.0)
    env.record_step([0.5, 0.75], 3.0)
    export_recording(env, success=True)
    export_recording(env, success=True)
    env.recorder_manager.record_post_reset([0])
    export_recording(env)
    env.record_step([-0.25, 0.0], 4.0)
    # A stale native success flag must not turn a manually interrupted attempt into a success.
    env.native_success.fill_(True)
    close_recording(env)

    with h5py.File(path, "r") as dataset:
        data = dataset["data"]
        assert list(data) == (["demo_0", "demo_1"] if export_mode == "all" else ["demo_0"])
        assert data.attrs["total"] == (3 if export_mode == "all" else 2)
        metadata = json.loads(data.attrs["env_args"])
        assert metadata["env_name"] == env.cfg.env_name
        assert metadata["sim_args"]["dt"] == 0.01
        assert metadata["sim_args"]["decimation"] == 2
        episode = data["demo_0"]
        assert bool(episode.attrs["success"])
        assert episode.attrs["num_samples"] == 2
        np.testing.assert_array_equal(episode["actions"][:], [[0.25, -0.5], [0.5, 0.75]])
        np.testing.assert_array_equal(episode["initial_state/rigid_object/object/root_pose"][:, 0], [1.0])
        np.testing.assert_array_equal(episode["states/rigid_object/object/root_pose"][:, 0], [2.0, 3.0])
        np.testing.assert_array_equal(episode["processed_actions"][:], [[7.0, 8.0], [7.0, 8.0]])
        np.testing.assert_array_equal(episode["obs"][:], [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        if export_mode == "all":
            assert not bool(data["demo_1"].attrs["success"])
            np.testing.assert_array_equal(data["demo_1/actions"][:], [[-0.25, 0.0]])


def test_registered_recorder_preserves_task_observation_layout_and_native_rules(recording_env):
    env, path = recording_env(recorder_cfg=StateOnlyRecorderCfg)
    assert env.cfg.observations.policy.concatenate_terms is True
    assert env.cfg.terminations.time_out is not None
    assert env.cfg.terminations.success is not None
    assert env.cfg.commands.resampling_time_range == (1.0, 2.0)
    env.record_step([0.25, 0.5], 2.0)
    # Use the native pre-reset hook, which must export before resetting the in-memory episode.
    env.native_success.fill_(True)
    env.recorder_manager.record_pre_reset([0])
    env.recorder_manager.reset([0])
    env.recorder_manager.record_post_reset([0])
    close_recording(env)

    with h5py.File(path, "r") as dataset:
        assert list(dataset["data"]) == ["demo_0"]
        episode = dataset["data/demo_0"]
        assert "obs" not in episode
        assert bool(episode.attrs["success"])
        np.testing.assert_array_equal(episode["actions"][:], [[0.25, 0.5]])


def test_configuration_rejects_overwrite_and_invalid_paths_without_mutating_task(tmp_path, recording_env):
    cfg = SimpleNamespace(recorders=object())
    original = cfg.recorders
    existing = tmp_path / "existing.hdf5"
    existing.write_bytes(b"existing recording")
    with pytest.raises(FileExistsError):
        configure_recording(cfg, "unused", existing, "all")
    assert existing.read_bytes() == b"existing recording"
    with pytest.raises(ValueError, match="hdf5"):
        configure_recording(cfg, "unused", tmp_path / "bad.json", "all")
    assert configure_recording(cfg, "unused", None, "success") is None
    assert cfg.recorders is original

    # The writer must use exactly the path that was checked, even with repeated suffixes.
    env, path = recording_env(filename="existing.hdf5.hdf5")
    close_recording(env)
    assert existing.read_bytes() == b"existing recording"
    assert path == tmp_path / "existing.hdf5.hdf5"
    with h5py.File(path, "r") as dataset:
        assert list(dataset["data"]) == []


def test_reset_after_success_skips_custom_summary_only_episode_and_restores_native_export(recording_env):
    env, path = recording_env(export_mode="all", recorder_cfg=StateOnlyRecorderCfg)
    env.record_step([0.25, 0.5], 2.0)
    export_recording(env, success=True)
    reset_after_export(env)
    assert env.recorder_manager.cfg.export_in_record_pre_reset is True

    # A later native reset must still export its actual episode, including a failed outcome.
    env.record_step([-0.25, -0.5], 3.0)
    env.reset()
    close_recording(env)

    with h5py.File(path, "r") as dataset:
        data = dataset["data"]
        assert list(data) == ["demo_0", "demo_1"]
        assert data.attrs["total"] == 2
        assert bool(data["demo_0"].attrs["success"])
        assert not bool(data["demo_1"].attrs["success"])
        assert data["demo_0"].attrs["num_samples"] == data["demo_1"].attrs["num_samples"] == 1
        np.testing.assert_array_equal(data["demo_0/terminal_position"][:], [[2.0, 2.0, 3.0]])
        np.testing.assert_array_equal(data["demo_1/terminal_position"][:], [[3.0, 2.0, 3.0]])
