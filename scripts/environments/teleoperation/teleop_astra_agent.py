# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Astra planning and visual action inference in the G1 fixed-base environment."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from astra_teleop.cli import normalize_cli_args
from astra_teleop.core import AstraConfig, AstraPipeline, MockClient

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Resolve the task and launch a single G1 controlled by two asynchronous Astra workers."""
    import warp as wp

    wp.config.enable_backward = False

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs import ManagerBasedRLEnvCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs")
    parser.add_argument(
        "--goal", default="Observe the scene. Hold both wrists in their current poses and keep both grippers open."
    )
    parser.add_argument("--model", default=os.environ.get("ASTRA_MODEL", "openai/openai/gpt-6-astra"))
    parser.add_argument(
        "--base_url",
        default=os.environ.get("ASTRA_BASE_URL", "https://inference-api.nvidia.com/v1"),
        help="Provider API base URL.",
    )
    parser.add_argument(
        "--api_format",
        choices=("responses", "chat_completions"),
        default=os.environ.get("ASTRA_API_FORMAT", "chat_completions"),
    )
    parser.add_argument("--api_key_env", default="NV_API_KEY", help="Environment variable containing the provider key.")
    parser.add_argument(
        "--api_key_file",
        type=Path,
        help="Read the provider key here; otherwise use the key environment variable or ~/.config/openai/api_key.",
    )
    parser.add_argument("--mock", action="store_true", help="Use local commands that hold wrists and open the hands.")
    parser.add_argument(
        "--pause_during_inference",
        action="store_true",
        help="Freeze physics while inference is pending and no usable action is ready; keep the viewport responsive.",
    )
    parser.add_argument(
        "--bin_drop", action="store_true", help="Expose bin-drop criteria and verify released, settled success locally."
    )
    parser.add_argument(
        "--reset_on_success",
        action="store_true",
        help="Reset after verified bin-drop success and start a fresh attempt.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=0, help="Stop after this many completed attempts; 0 is unlimited."
    )
    parser.add_argument("--system1_effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--system2_effort", choices=("low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--system1_interval", type=float, default=1.0, help="Minimum action request interval [s].")
    parser.add_argument("--system2_interval", type=float, default=10.0, help="Minimum planning request interval [s].")
    parser.add_argument("--system1_timeout", type=float, default=20.0, help="Action request timeout [s].")
    parser.add_argument("--system2_timeout", type=float, default=60.0, help="Planning request timeout [s].")
    parser.add_argument("--system1_max_age", type=float, default=10.0, help="Maximum action observation age [s].")
    parser.add_argument("--system2_max_age", type=float, default=30.0, help="Maximum planner observation age [s].")
    parser.add_argument("--max_hold_seconds", type=float, default=1.0, help="Maximum time to execute one command [s].")
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=1,
        help="Waypoints per inference request (1 keeps single-command mode).",
    )
    parser.add_argument(
        "--waypoint_seconds", type=float, default=0.5, help="Simulation seconds per waypoint in chunk mode."
    )
    parser.add_argument("--wrist_linear_speed", type=float, default=0.1, help="Wrist setpoint speed limit [m/s].")
    parser.add_argument("--wrist_angular_speed", type=float, default=0.8, help="Wrist setpoint rotation limit [rad/s].")
    parser.add_argument("--grip_speed", type=float, default=1.6, help="Finger joint setpoint speed limit [rad/s].")
    parser.add_argument(
        "--astra_log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Astra console detail, independent of simulator logging. DEBUG includes each waypoint target.",
    )
    parser.add_argument(
        "--status_interval", type=float, default=5.0, help="Wall seconds between progress summaries; 0 disables them."
    )
    parser.add_argument(
        "--cameras",
        default="robot_pov_cam,left_wrist_cam,right_wrist_cam",
        help="Comma-separated RGB sensors sent to Astra; selecting a wrist camera adds it to the scene.",
    )
    parser.add_argument("--image_size", type=int, default=640, help="Maximum image edge in pixels.")
    parser.add_argument(
        "--image_trace_dir", type=Path, help="Save the exact JPEG views sent to inference for camera debugging."
    )
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many simulation steps; 0 runs until exit."
    )
    parser.add_argument(
        "--trace", type=Path, help="Write JSONL state, inference events, and applied actions (no images/key)."
    )
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, normalize_cli_args(argv))
    if args.image_size < 1 or args.max_steps < 0 or args.max_episodes < 0:
        parser.error("image_size must be positive; max_steps and max_episodes must be nonnegative")
    if args.reset_on_success and not args.bin_drop:
        parser.error("--reset_on_success requires --bin_drop so success can be verified independently")
    if not math.isfinite(args.status_interval) or args.status_interval < 0:
        parser.error("status_interval must be finite and nonnegative")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.wrist_linear_speed, args.wrist_angular_speed, args.grip_speed)
    ):
        parser.error("wrist and grip speed limits must be positive and finite")
    _configure_logging(args.astra_log_level)
    args.cameras = [name.strip() for name in args.cameras.split(",")]
    if not all(args.cameras) or len(set(args.cameras)) != len(args.cameras):
        parser.error("cameras must be a comma-separated list of unique sensor names")

    api_key = ""
    if not args.mock:
        if not args.base_url:
            parser.error("Set --base_url or ASTRA_BASE_URL to the API endpoint that issued your key")
        api_key = os.environ.get(args.api_key_env, "").strip() if args.api_key_file is None else ""
        key_file = args.api_key_file or Path.home() / ".config/openai/api_key"
        if args.api_key_file is not None or (not api_key and key_file.is_file()):
            try:
                api_key = key_file.expanduser().read_text().strip()
            except OSError:
                parser.error(f"Cannot read API key file {key_file}; run scripts/tools/configure_astra_key.py")
        if not api_key:
            parser.error(
                f"No API key found. Run scripts/tools/configure_astra_key.py, set {args.api_key_env}, "
                "or pass --api_key_file. Use --mock for local validation."
            )
    config = AstraConfig(
        goal=args.goal,
        model=args.model,
        base_url=args.base_url or "https://inference-api.nvidia.com/v1",
        api_format=args.api_format,
        api_key=api_key,
        system1_effort=args.system1_effort,
        system2_effort=args.system2_effort,
        system1_interval=args.system1_interval,
        system2_interval=args.system2_interval,
        system1_timeout=args.system1_timeout,
        system2_timeout=args.system2_timeout,
        system1_max_age=args.system1_max_age,
        system2_max_age=args.system2_max_age,
        max_hold_seconds=args.max_hold_seconds,
        action_chunk_size=args.action_chunk_size,
        waypoint_seconds=args.waypoint_seconds,
    )
    env_cfg, _ = resolve_task_config(args.task, "", overrides=overrides)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("Astra teleoperation requires a manager-based RL environment.")
    from astra_teleop.cameras import add_wrist_cameras
    from astra_teleop.perception import configure_object_contact_sensor

    add_wrist_cameras(env_cfg.scene, args.cameras)
    configure_object_contact_sensor(env_cfg.scene)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    # One render per control step is enough for the viewport and inference
    # snapshots; keep the physics substeps and action cadence unchanged.
    env_cfg.sim.render_interval = max(env_cfg.sim.render_interval, env_cfg.decimation)
    env_cfg.env_name = args.task
    env_cfg.terminations.time_out = None
    args.original_success_cfg = env_cfg.terminations.success
    # The task's training success checks only an object-root box. It can reset
    # before release or a completed handoff; let Astra assess the requested goal.
    env_cfg.terminations.success = None
    # The fixed-base task does not observe the camera itself; refresh its RTX state after resets.
    env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 3)
    args.enable_cameras = True
    args.require_kit = True

    with launch_simulation(env_cfg, args), ExitStack() as cleanup:
        import gymnasium as gym

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        cleanup.callback(env.close)
        pipeline = AstraPipeline(config, client=MockClient() if args.mock else None)
        cleanup.callback(pipeline.close)
        trace = None
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = cleanup.enter_context(args.trace.open("w", encoding="utf-8"))
        run(env, pipeline, args, trace)


def _configure_logging(level: str) -> None:
    """Keep Astra progress visible without enabling verbose Kit or HTTP logging."""
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s Astra %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)


def _log_events(events: list[dict]) -> None:
    """Summarize lifecycle events without logging request bodies or camera payloads."""
    for event in events:
        kind = event["event"]
        role = event.get("role", "")
        latency = event.get("latency_seconds")
        timing = "" if latency is None else f" after {latency:.2f}s"
        if kind == "submitted":
            logger.info("%s request submitted.", role)
        elif kind in {"inference_paused", "inference_resumed"}:
            logger.info("Physics %s at step %d (sim %.2fs).", kind.split("_")[1], event["step"], event["sim_time"])
        elif kind == "accepted":
            result = event["result"]
            if role == "system2":
                logger.info("Plan%s: %s (stop=%s)", timing, result["subgoal"], result["stop"])
                logger.debug("Plan success criteria: %s", result["success_criteria"])
            else:
                logger.info(
                    "system1 reply%s: %d waypoint(s), stop=%s",
                    timing,
                    len(result["commands"]) if "commands" in result else 1,
                    result["stop"],
                )
        elif kind == "error":
            logger.warning("%s request failed%s: %s", role, timing, event["error"])
        elif kind == "discarded":
            logger.warning("%s reply discarded%s: %s", role, timing, event["reason"])
        elif kind == "underrun":
            logger.info("Action buffer empty; holding wrists and retaining grip targets while awaiting inference.")
        elif kind == "plan_expired":
            logger.info("Plan expired; cleared action buffer and holding until a fresh plan arrives.")
        elif kind == "feedback_checkpoint":
            logger.info(
                "Chunk %s paused after %d waypoint(s); requesting a fresh visual assessment.",
                event["chunk_id"],
                event["completed_waypoints"],
            )
        elif kind == "feedback_ready":
            logger.info(
                "Fresh assessment received after chunk %s; %s.",
                event["chunk_id"],
                "stop requested" if event["stop"] else "action inference can resume",
            )
        elif kind in {"chunk_started", "chunk_queued", "chunk_completed"}:
            logger.info("%s: chunk %s", kind.replace("_", " "), event["chunk_id"])
        elif kind == "reset":
            logger.info("Episode %s: cleared plans and actions.", event["episode"])
        else:
            logger.debug("Pipeline event: %s", event)


class _EpisodeMetrics:
    """Keep per-attempt timings separate from late replies and reset/setup overhead."""

    def __init__(self, step: int):
        self.started = time.monotonic()
        self.start_step = step
        self.first_action_seconds = None
        self.latencies = {"system1": [], "system2": []}
        self.requests = {"system1": 0, "system2": 0}
        self.latest_previous_reply_seconds = None
        self.feedback_count = 0

    def record_action(self):
        if self.first_action_seconds is None:
            self.first_action_seconds = time.monotonic() - self.started

    def record_events(self, events: list[dict]):
        for event in events:
            if event["event"] == "submitted":
                self.requests[event["role"]] += 1
            elif event["event"] == "accepted":
                self.latencies[event["role"]].append(event["latency_seconds"])
            elif (
                event["event"] == "discarded"
                and "latency_seconds" in event
                and event["time"] - event["latency_seconds"] < self.started
            ):
                self.latest_previous_reply_seconds = event["time"] - self.started
            elif event["event"] == "feedback_checkpoint":
                self.feedback_count += 1

    def finish(self, episode_id, step, step_dt, verified_success, task_progress, boundary_events, trace):
        self.record_events(boundary_events)
        _log_events(boundary_events)
        outcome = {
            "event": "episode_completed",
            "episode_id": episode_id,
            "outcome": "verified_bin_drop" if verified_success else "environment_termination",
            "wall_seconds": time.monotonic() - self.started,
            "simulation_seconds": (step - self.start_step) * step_dt,
            "ready_to_first_action_seconds": self.first_action_seconds,
            "accepted_request_latency_seconds": self.latencies,
            "requests_submitted": self.requests,
            "latest_previous_episode_reply_seconds": self.latest_previous_reply_seconds,
            "feedback_checkpoints": self.feedback_count,
            "task_progress": task_progress if verified_success else None,
            "cross_episode_policy_memory": False,
        }
        logger.info("Episode outcome: %s", json.dumps(outcome, allow_nan=False))
        if trace:
            row = {"step": step, "episode_id": episode_id, "events": boundary_events + [outcome]}
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            trace.flush()


def _log_task_progress(progress):
    if progress is None:
        return
    measured = progress["measurements"]
    logger.info(
        "Bin: drop-ready=%s | below-wall XY margin %s | tipping margin %.3fm | floor gap %.3fm "
        "| robot contact %sN | bin support %sN "
        "| hands open=%s | settled %.2fs",
        progress["drop_ready"],
        "above wall" if measured["below_wall_xy_margin_m"] is None else f"{measured['below_wall_xy_margin_m']:+.3f}m",
        measured["rotation_safe_xy_margin_m"],
        measured["floor_gap_m"],
        "unknown"
        if measured["robot_contact_magnitude_sum_n"] is None
        else f"{measured['robot_contact_magnitude_sum_n']:.4g}",
        "unknown" if measured["bin_upward_support_n"] is None else f"{measured['bin_upward_support_n']:.2f}",
        progress["conditions"]["both_grippers_open"],
        progress["settled_sim_seconds"],
    )


def run(env, pipeline: AstraPipeline, args: argparse.Namespace, trace) -> None:
    """Step at the environment control rate while inference uses copied snapshots in worker threads."""
    import torch
    from astra_teleop.g1 import G1ActionAdapter
    from astra_teleop.motion import WristTrajectory
    from astra_teleop.perception import BinDropObserver, ObjectPerception

    with torch.inference_mode():
        observations, _ = env.reset()
        perception = ObjectPerception(env) if "object_hand_contact" in env.scene.sensors else None
        adapter = G1ActionAdapter(env, perception=perception, grip_speed=args.grip_speed)
        task_observer = BinDropObserver(env, perception, adapter, args.original_success_cfg) if args.bin_drop else None
        task_progress = task_observer.update(0.0) if task_observer is not None else None
        action_spec = adapter.describe()
        if task_observer is not None:
            action_spec["task_contract"] = task_observer.task.describe()
            logger.info("Loaded environment-derived bin-drop contract: %s", json.dumps(action_spec["task_contract"]))
        action_spec["limits"].update(
            wrist_linear_speed_m_s=args.wrist_linear_speed,
            wrist_angular_speed_rad_s=args.wrist_angular_speed,
        )
        action_spec["execution_timing"] = {
            "pause_during_inference": args.pause_during_inference,
            "waypoint_clock": "Chunk dwells use executed simulation time; legacy single-command expiry uses wall time.",
            "inference_wait": (
                "Physics freezes while a request is pending and no command is ready. Waiting for a reply provides no "
                "settling, finger motion, or stable-hold duration. Schedule explicit unchanged waypoints for required "
                "closure, settling, and stability checks before requesting feedback."
                if args.pause_during_inference
                else "Physics and accepted grip trajectories continue during ordinary valid-plan inference waits."
            ),
        }
        episode_id = 0
        step = 0
        capture_sequence = 0
        command_id = None
        previous_plan = None
        inference_paused = False
        episode_metrics = _EpisodeMetrics(step)
        applied_action = adapter.idle_action()
        idle_holding = True
        executed_action = applied_action
        wrist_motion = WristTrajectory(
            applied_action, linear_speed=args.wrist_linear_speed, angular_speed=args.wrist_angular_speed
        )
        deadline = time.monotonic()
        status_time = deadline
        status_step = 0
        logger.info(
            "Started %s control; model=%s; cameras=%s; control dt=%.3fs.",
            "mock" if args.mock else "live",
            pipeline.config.model,
            ",".join(args.cameras),
            env.step_dt,
        )
        logger.info("Goal: %s", pipeline.config.goal)
        logger.info(
            "Motion limits: wrists %.3f m/s, %.3f rad/s; fingers %.3f rad/s.",
            args.wrist_linear_speed,
            args.wrist_angular_speed,
            args.grip_speed,
        )
        if pipeline.config.action_chunk_size > 1:
            logger.info(
                "Chunking: %d waypoints x %.2fs = %.2f simulation seconds; one next chunk prefetched.",
                pipeline.config.action_chunk_size,
                pipeline.config.waypoint_seconds,
                pipeline.config.action_chunk_size * pipeline.config.waypoint_seconds,
            )
        if args.trace:
            logger.info("Trace: %s", args.trace.resolve())
        if args.image_trace_dir:
            args.image_trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Inference camera images: %s", args.image_trace_dir.resolve())
        while env.sim.is_running() and (args.max_steps == 0 or step < args.max_steps):
            loop_started = time.monotonic()
            snapshot = None
            images = None
            capture_id = None
            if pipeline.needs_snapshot():
                capture_sequence += 1
                capture_id = capture_sequence
                snapshot = adapter.snapshot(observations, episode_id, step, step * env.step_dt)
                if task_progress is not None:
                    snapshot["task_progress"] = task_progress
                images = adapter.capture_images(args.cameras, args.image_size)
                if args.image_trace_dir:
                    for view in images:
                        path = args.image_trace_dir / (
                            f"episode-{episode_id:04d}-step-{step:08d}-capture-{capture_id:06d}"
                            f"-{Path(view['name']).name}.jpg"
                        )
                        path.write_bytes(base64.b64decode(view["data_url"].split(",", 1)[1]))
                    logger.debug("Saved %d camera views at step %d.", len(images), step)
            command = pipeline.poll(snapshot=snapshot, images=images, action_spec=action_spec)
            status = pipeline.status
            paused = args.pause_during_inference and command is None and bool(status["in_flight"])
            if not paused:
                plan_changed = status["plan"] != previous_plan
                previous_plan = status["plan"]
                if command is None:
                    if (
                        pipeline.config.action_chunk_size > 1
                        and status["plan"] is not None
                        and not status["plan"]["stop"]
                    ):
                        idle_holding = False
                        if not plan_changed and command_id is not None:
                            # End the allocated wrist motion at its last setpoint, retaining a stable pose hold.
                            applied_action = applied_action.clone()
                            applied_action[:, :14] = executed_action[:, :14]
                        # Reanchor a preempted wrist path once; ordinary waits retain accepted targets.
                        applied_action = (
                            adapter.hold_action(env.step_dt)
                            if plan_changed
                            else adapter.advance_grip(applied_action, env.step_dt)
                        )
                        if plan_changed:
                            wrist_motion.reset(applied_action)
                    else:
                        if not idle_holding:
                            # Cancel motion once, then hold that measured pose instead of following drift.
                            applied_action = adapter.idle_action()
                            wrist_motion.reset(applied_action)
                            idle_holding = True
                    command_id = None
                else:
                    try:
                        if status["command_id"] != command_id:
                            applied_action = adapter.action(command)
                            idle_holding = False
                            command_id = status["command_id"]
                            episode_metrics.record_action()
                            logger.debug(
                                "Applying command %s, chunk=%s waypoint=%s; targets=%s",
                                command_id,
                                status["chunk_id"],
                                status["waypoint_index"],
                                command,
                            )
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "Wrist targets after adapter bounds (left then right, position + XYZW): %s",
                                    applied_action[0, :14].cpu().tolist(),
                                )
                        applied_action = adapter.advance_grip(applied_action, env.step_dt)
                    except ValueError:
                        logger.warning("Astra command rejected by G1 adapter; holding and requesting a new plan.")
                        pipeline.reset()
                        applied_action = adapter.idle_action()
                        wrist_motion.reset(applied_action)
                        idle_holding = True
                        command_id = None
                executed_action = wrist_motion.advance(applied_action, env.step_dt)
            events = pipeline.drain_events()
            episode_metrics.record_events(events)
            if paused != inference_paused:
                events.append(
                    {
                        "event": "inference_paused" if paused else "inference_resumed",
                        "step": step,
                        "sim_time": step * env.step_dt,
                    }
                )
                inference_paused = paused
                deadline = time.monotonic()
            _log_events(events)
            now = time.monotonic()
            if args.status_interval and now - status_time >= args.status_interval:
                status = pipeline.status
                logger.info(
                    "Step %d | %.1f steps/s | sim %.2fs | %s | chunk=%s waypoint=%s/%d next=%s | requests=%s",
                    step,
                    (step - status_step) / (now - status_time),
                    step * env.step_dt,
                    "paused for inference" if paused else "holding" if command_id is None else "executing",
                    status["chunk_id"],
                    "-" if status["waypoint_index"] is None else status["waypoint_index"] + 1,
                    status["chunk_size"],
                    status["queued_chunk_id"],
                    ",".join(status["in_flight"]) or "none",
                )
                status_time, status_step = now, step
                _log_task_progress(task_progress)
            if trace and (not paused or snapshot is not None or events):
                trace.write(
                    json.dumps(
                        {
                            "step": step,
                            "episode_id": episode_id,
                            "snapshot": snapshot,
                            "capture_id": capture_id,
                            "events": events,
                            "status": pipeline.status,
                            "inference_paused": paused,
                            "task_progress": task_progress,
                            "action": None if paused else executed_action[0].cpu().tolist(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                trace.flush()
            if paused:
                # KitVisualizer pumps UI updates with physics disabled. Do not advance
                # adapter trajectories or command dwell clocks during this render-only wait.
                env.sim.render()
                time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - loop_started)))
                deadline = time.monotonic()
                continue
            observations, _, terminated, truncated, _ = env.step(executed_action)
            pipeline.advance(env.step_dt)
            step += 1
            environment_reset = bool((terminated | truncated).any())
            if task_observer is not None and not environment_reset:
                task_progress = task_observer.update(env.step_dt)
            verified_success = not environment_reset and task_progress is not None and task_progress["success"]
            if verified_success or environment_reset:
                # advance() may have completed a checkpoint on the final physics step.
                # Attribute those events before resetting the pipeline or ending the run.
                episode_metrics.finish(
                    episode_id, step, env.step_dt, verified_success, task_progress, pipeline.drain_events(), trace
                )
                if (verified_success and not args.reset_on_success) or (
                    args.max_episodes and episode_id + 1 >= args.max_episodes
                ):
                    break
                if verified_success:
                    observations, _ = env.reset()
                # A termination already reset the environment; invalidate all preceding inference.
                episode_id += 1
                pipeline.reset()
                adapter.reset()
                if task_observer is not None:
                    task_observer.reset()
                    task_progress = task_observer.update(0.0)
                applied_action = adapter.idle_action()
                idle_holding = True
                executed_action = applied_action
                wrist_motion.reset(applied_action)
                command_id = None
                previous_plan = None
                episode_metrics = _EpisodeMetrics(step)
            deadline += env.step_dt
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
        logger.info("Stopped after %d control steps (%.2f simulation seconds).", step, step * env.step_dt)


if __name__ == "__main__":
    main()
