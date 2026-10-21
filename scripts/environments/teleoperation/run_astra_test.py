# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test an Isaac Lab environment with an explicit Astra action/observation adapter."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import math
import os
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from astra_teleop.cli import normalize_cli_args
from astra_teleop.core import AstraConfig, AstraPipeline, MockClient

logger = logging.getLogger("astra_testbed")


def main(argv: list[str] | None = None) -> None:
    """Load a registered task and its adapter, retaining the task's native episode rules."""
    argv = normalize_cli_args(argv)
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs import ManagerBasedRLEnvCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    # Registration must precede preset discovery, including --task=X --help.
    registrations = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    registrations.add_argument("--env_module", action="append", default=[])
    early, _ = registrations.parse_known_args(argv)
    for name in early.env_module:
        importlib.import_module(name)

    parser = argparse.ArgumentParser(description=__doc__, parents=[registrations])
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--adapter", required=True, help="Importable module:AdapterClass implementing EnvironmentAdapter."
    )
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model", default=os.environ.get("ASTRA_MODEL", "openai/openai/gpt-6-astra"))
    parser.add_argument("--base_url", default=os.environ.get("ASTRA_BASE_URL", "https://inference-api.nvidia.com/v1"))
    parser.add_argument(
        "--api_format",
        choices=("responses", "chat_completions"),
        default=os.environ.get("ASTRA_API_FORMAT", "chat_completions"),
    )
    parser.add_argument("--api_key_env", default="NV_API_KEY")
    parser.add_argument("--api_key_file", type=Path)
    parser.add_argument(
        "--mock", action="store_true", help="Use the adapter's neutral command without external inference."
    )
    parser.add_argument("--pause_during_inference", action="store_true")
    parser.add_argument("--reset_on_success", action="store_true", help="Repeat after adapter-verified success.")
    parser.add_argument("--max_steps", type=int, default=0, help="Total executed control steps; 0 is unlimited.")
    parser.add_argument(
        "--max_episodes", type=int, default=0, help="Completed attempts, including failures; 0 is unlimited."
    )
    parser.add_argument("--cameras", default="", help="Comma-separated RGB sensor names; empty selects no cameras.")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--image_trace_dir", type=Path)
    parser.add_argument("--astra_log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--status_interval", type=float, default=5.0)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("logs/astra") / f"testbed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.jsonl",
    )
    parser.add_argument("--system1_effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--system2_effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--system1_interval", type=float, default=1.0)
    parser.add_argument("--system2_interval", type=float, default=300.0)
    parser.add_argument("--system1_timeout", type=float, default=75.0)
    parser.add_argument("--system2_timeout", type=float, default=180.0)
    parser.add_argument("--system1_max_age", type=float, default=120.0)
    parser.add_argument("--system2_max_age", type=float, default=360.0)
    parser.add_argument("--max_hold_seconds", type=float, default=1.0)
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--waypoint_seconds", type=float, default=0.1)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if args.max_steps < 0 or args.max_episodes < 0 or args.image_size < 1:
        parser.error("max_steps/max_episodes must be nonnegative and image_size positive")
    if not math.isfinite(args.status_interval) or args.status_interval < 0:
        parser.error("status_interval must be finite and nonnegative")
    args.cameras = [name.strip() for name in args.cameras.split(",")] if args.cameras.strip() else []
    if any(not name for name in args.cameras) or len(args.cameras) != len(set(args.cameras)):
        parser.error("cameras must contain unique, nonempty sensor names")
    module_name, separator, class_name = args.adapter.partition(":")
    if not separator or not module_name or not class_name.isidentifier():
        parser.error("adapter must have the form importable.module:AdapterClass")
    adapter_class = getattr(importlib.import_module(module_name), class_name)

    api_key = ""
    if not args.mock:
        api_key = os.environ.get(args.api_key_env, "").strip() if args.api_key_file is None else ""
        key_file = (args.api_key_file or Path.home() / ".config/openai/api_key").expanduser()
        if args.api_key_file is not None or (not api_key and key_file.is_file()):
            try:
                api_key = key_file.read_text().strip()
            except OSError:
                parser.error(f"Cannot read {key_file}; run scripts/tools/configure_astra_key.py")
        if not api_key:
            parser.error(f"Set {args.api_key_env} or --api_key_file, or use --mock")
    config_names = (
        "goal",
        "model",
        "base_url",
        "api_format",
        "system1_effort",
        "system2_effort",
        "system1_interval",
        "system2_interval",
        "system1_timeout",
        "system2_timeout",
        "system1_max_age",
        "system2_max_age",
        "max_hold_seconds",
        "action_chunk_size",
        "waypoint_seconds",
    )
    try:
        config = AstraConfig(api_key=api_key, **{name: getattr(args, name) for name in config_names})
    except ValueError as error:
        parser.error(str(error))
    logger.setLevel(args.astra_log_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s Astra %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)

    env_cfg, _ = resolve_task_config(args.task, "", overrides=overrides)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("The testbed runner currently supports ManagerBasedRLEnv tasks.")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.env_name = args.task
    args.enable_cameras = bool(args.cameras)
    adapter_class.configure_env(env_cfg, args)
    with launch_simulation(env_cfg, args), ExitStack() as cleanup:
        import gymnasium as gym
        import torch

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        cleanup.callback(env.close)
        if env.num_envs != 1:
            raise ValueError("The Astra testbed requires exactly one environment.")
        with torch.inference_mode():
            observations, _ = env.reset()
            adapter = adapter_class(env)
            contract = adapter.contract
            pipeline = AstraPipeline(
                config, client=MockClient(action_contract=contract) if args.mock else None, action_contract=contract
            )
            cleanup.callback(pipeline.close)
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = cleanup.enter_context(args.trace.open("w", encoding="utf-8"))
            _write(
                trace,
                {
                    "event": "run_started",
                    "task": args.task,
                    "adapter": args.adapter,
                    "mock": args.mock,
                    "inference": {name: getattr(config, name) for name in config_names},
                },
            )
            run(env, pipeline, adapter, observations, args, trace)


def _write(trace, row: dict) -> None:
    trace.write(json.dumps(row, allow_nan=False) + "\n")
    trace.flush()


def _record_events(events, latencies) -> None:
    for event in events:
        kind = event["event"]
        role = event.get("role", "")
        if kind == "accepted":
            latencies[role].append(event["latency_seconds"])
            result = event["result"]
            logger.info("%s reply %.2fs: %s", role, event["latency_seconds"], result.get("subgoal", "actions ready"))
            logger.debug("Accepted %s: %s", role, result)
        elif kind in {"error", "discarded"}:
            logger.warning("%s %s: %s", role, kind, event.get("error", event.get("reason")))
        else:
            logger.info("%s %s", role, kind)


def run(env, pipeline, adapter, observations, args, trace) -> None:
    """Execute adapter-owned commands; workers only receive detached snapshots and encoded images."""
    action_spec = adapter.describe()
    action_spec["execution_timing"] = {
        "pause_during_inference": args.pause_during_inference,
        "waypoint_clock": (
            "Executed simulation time, rounded up to a control step; inference pauses do not advance physics."
            if args.action_chunk_size > 1
            else "Single-command hold uses wall time; inference pauses do not advance physics."
        ),
    }
    _write(
        trace,
        {
            "event": "action_spec",
            "description": action_spec,
            "command_schema": adapter.contract.command_schema,
            "instructions": adapter.contract.instructions,
        },
    )
    progress = adapter.task_progress(0.0)
    episode_id = step = capture_id = 0
    episode_start_step = 0
    episode_started = status_time = time.monotonic()
    first_action_seconds = None
    latencies = {"system1": [], "system2": []}
    was_paused = False
    outcome = "window_closed"
    if args.image_trace_dir:
        args.image_trace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Task=%s adapter=%s mode=%s trace=%s", args.task, args.adapter, "mock" if args.mock else "live", args.trace
    )
    while env.sim.is_running():
        if args.max_steps and step >= args.max_steps:
            outcome = "max_steps"
            break
        loop_started = time.monotonic()
        snapshot = images = None
        if pipeline.needs_snapshot():
            capture_id += 1
            snapshot = adapter.snapshot(observations, episode_id, step, step * env.step_dt)
            if progress is not None:
                snapshot["task_progress"] = progress
            images = adapter.capture_images(args.cameras, args.image_size)
            if args.image_trace_dir:
                for view in images:
                    image_path = args.image_trace_dir / (
                        f"episode-{episode_id:04d}-step-{step:08d}-capture-{capture_id:06d}-{Path(view['name']).name}.jpg"
                    )
                    image_path.write_bytes(base64.b64decode(view["data_url"].split(",", 1)[1]))
        command = pipeline.poll(snapshot=snapshot, images=images, action_spec=action_spec)
        status = pipeline.status
        paused = args.pause_during_inference and command is None and bool(status["in_flight"])
        action = None
        if not paused:
            try:
                action = adapter.idle_action() if command is None else adapter.action(command)
                if command is not None and first_action_seconds is None:
                    first_action_seconds = time.monotonic() - episode_started
            except ValueError as error:
                logger.warning("Adapter rejected command: %s; resetting inference", error)
                pipeline.reset()
                action = adapter.idle_action()
        events = pipeline.drain_events()
        _record_events(events, latencies)
        if paused != was_paused:
            logger.info("Physics %s at step %d", "paused for inference" if paused else "resumed", step)
            events.append({"event": "inference_paused" if paused else "inference_resumed"})
            was_paused = paused
        if args.status_interval and time.monotonic() - status_time >= args.status_interval:
            logger.info(
                "Step %d sim %.2fs episode %d pending=%s progress=%s",
                step,
                step * env.step_dt,
                episode_id,
                status["in_flight"],
                progress,
            )
            status_time = time.monotonic()
        if not paused or snapshot is not None or events:
            _write(
                trace,
                {
                    "step": step,
                    "episode_id": episode_id,
                    "capture_id": capture_id,
                    "snapshot": snapshot,
                    "events": events,
                    "status": status,
                    "task_progress": progress,
                    "inference_paused": paused,
                    "action": None if action is None else action.detach().cpu().tolist(),
                },
            )
        if paused:
            env.sim.render()
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - loop_started)))
            continue
        observations, _, terminated, truncated, _ = env.step(action)
        pipeline.advance(env.step_dt)
        step += 1
        environment_reset = bool((terminated | truncated).any())
        # ManagerBasedRLEnv already reset done environments. Never interpret the
        # reset scene as a terminal measurement from the preceding attempt.
        progress = None if environment_reset else adapter.task_progress(env.step_dt)
        success = progress is not None and progress["success"] is True
        if success or environment_reset:
            boundary_events = pipeline.drain_events()
            _record_events(boundary_events, latencies)
            outcome = "verified_success" if success else "environment_termination"
            result = {
                "event": "episode_completed",
                "episode_id": episode_id,
                "outcome": outcome,
                "terminated": bool(terminated.any()),
                "truncated": bool(truncated.any()),
                "wall_seconds": time.monotonic() - episode_started,
                "simulation_seconds": (step - episode_start_step) * env.step_dt,
                "ready_to_first_action_seconds": first_action_seconds,
                "accepted_request_latency_seconds": latencies,
                "task_progress": progress,
                "events": boundary_events,
                "cross_episode_policy_memory": False,
            }
            _write(trace, result)
            logger.info("Episode outcome: %s", json.dumps(result, allow_nan=False))
            if (success and not args.reset_on_success) or (args.max_episodes and episode_id + 1 >= args.max_episodes):
                break
            if success:
                observations, _ = env.reset()
            pipeline.reset()
            adapter.reset()
            progress = adapter.task_progress(0.0)
            episode_id += 1
            episode_start_step = step
            episode_started = time.monotonic()
            first_action_seconds = None
            latencies = {"system1": [], "system2": []}
            outcome = "window_closed"
        time.sleep(max(0.0, env.step_dt - (time.monotonic() - loop_started)))
    _write(trace, {"event": "run_completed", "outcome": outcome, "step": step, "episode_id": episode_id})
    logger.info("Stopped after %d control steps: %s", step, outcome)


if __name__ == "__main__":
    main()
