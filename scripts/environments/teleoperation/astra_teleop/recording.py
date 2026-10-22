# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record Astra executions through Isaac Lab's standard demonstration recorder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


def add_recording_args(parser: argparse.ArgumentParser) -> None:
    """Add optional HDF5 recording without changing the default execution path."""
    parser.add_argument(
        "--dataset_file", type=Path, help="Record demonstrations to a new HDF5 file; disabled when omitted."
    )
    parser.add_argument(
        "--dataset_export_mode",
        choices=("success", "all"),
        default="success",
        help="Keep only measured successes (default), or all attempts including interrupted episodes.",
    )


def configure_recording(
    env_cfg: ManagerBasedRLEnvCfg, task: str, dataset_file: Path | None, export_mode: str
) -> Path | None:
    """Select the task's demo recorder while preserving observations and native episode rules."""
    if dataset_file is None:
        if export_mode != "success":
            raise ValueError("--dataset_export_mode requires --dataset_file")
        return None
    if export_mode not in {"success", "all"}:
        raise ValueError("dataset_export_mode must be success or all")
    path = dataset_file.expanduser().resolve()
    if not path.suffix:
        path = path.with_suffix(".hdf5")
    if path.suffix != ".hdf5":
        raise ValueError("--dataset_file must end in .hdf5 or have no extension")
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}. Choose a new --dataset_file.")

    import gymnasium as gym

    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    entry_point = gym.spec(task.split(":")[-1]).kwargs.get("demo_recorder_cfg_entry_point")
    recorder = (
        ActionStateRecorderManagerCfg()
        if entry_point is None
        else load_cfg_from_registry(task, "demo_recorder_cfg_entry_point")
    )
    recorder.dataset_export_dir_path = str(path.parent)
    # The HDF5 writer accepts the suffix; stripping it can redirect names like demo.hdf5.hdf5.
    recorder.dataset_filename = path.name
    recorder.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_ONLY if export_mode == "success" else DatasetExportMode.EXPORT_ALL
    )
    # Native autoresets must export before the environment clears the episode buffer.
    recorder.export_in_record_pre_reset = True
    # Finalization below skips empty attempts and explicitly labels interrupted attempts.
    recorder.export_in_close = False
    env_cfg.recorders = recorder
    return path


def export_recording(env: ManagerBasedRLEnv, *, success: bool = False) -> None:
    """Finalize a measured success or incomplete attempt without creating empty demonstrations."""
    import torch

    recorder = env.recorder_manager
    if len(recorder.get_episode(0).data.get("actions", [])) == 0:
        return
    recorder.record_pre_reset([0], force_export_or_skip=False)
    # pre_reset reads native termination flags; apply Astra's measured outcome afterward.
    recorder.set_success_to_episodes([0], torch.tensor([[success]], dtype=torch.bool, device=env.device))
    recorder.export_episodes([0])


def close_recording(env: ManagerBasedRLEnv) -> None:
    """Flush a nonempty interrupted attempt according to the export mode, then close the file."""
    try:
        export_recording(env)
    finally:
        env.recorder_manager.close()


def reset_after_export(env: ManagerBasedRLEnv) -> tuple[dict, dict]:
    """Reset a finalized episode without exporting a custom pre-reset term a second time."""
    recorder = env.recorder_manager
    export_on_reset = recorder.cfg.export_in_record_pre_reset
    recorder.cfg.export_in_record_pre_reset = False
    try:
        return env.reset()
    finally:
        recorder.cfg.export_in_record_pre_reset = export_on_reset
