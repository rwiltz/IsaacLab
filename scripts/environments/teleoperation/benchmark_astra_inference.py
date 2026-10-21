# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a saved scene to compare two-stage Astra inference with a merged plan/action call.

Example (12 HTTP calls maximum; no simulator):
    python benchmark_astra_inference.py --fixture /tmp/astra-latency-fixture.json \
        --output /tmp/astra-latency.jsonl --api_key_file ~/.config/openai/api_key

The fixture contains snapshot, action_spec, goal, plan, and image_paths [{name, path}].
All variants see the same scene. The two-stage condition passes its newly generated
plan to System 1; merged conditions generate that plan and its actions in one reply.
Results are experimental latency measurements, not evidence of successful execution.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

from astra_teleop.core import (
    PLAN_SCHEMA,
    AstraConfig,
    AstraHTTPClient,
    _action_schema,
    _NoRedirect,
    _object_schema,
    _validate,
    _validate_result,
)


class _RequestCaptured(Exception):
    pass


def production_request(config, role, fixture, images, plan):
    """Capture the unmodified production request immediately before its HTTP call."""
    captured = {}

    def capture(opener, request, **kwargs):
        captured["request"] = request
        raise _RequestCaptured

    client = AstraHTTPClient(config)
    # This standalone benchmark is synchronous. The patch never spans a network
    # request or touches the running controller in another process.
    with patch("urllib.request.OpenerDirector.open", capture), suppress(_RequestCaptured):
        client.infer(role, fixture["snapshot"], images, fixture["action_spec"], plan)
    return captured["request"]


def build_request(config, variant, role, fixture, images, plan):
    """Reuse production role prompts and schemas; merge only for the experimental conditions."""
    if variant == "two_stage":
        return production_request(config, role, fixture, images, plan)
    planner = production_request(config, "system2", fixture, images, fixture.get("plan"))
    actor = production_request(config, "system1", fixture, images, fixture.get("plan"))
    payload, action_payload = json.loads(planner.data), json.loads(actor.data)
    is_chat = config.api_format == "chat_completions"
    planner_text = payload["messages"][0]["content"] if is_chat else payload["instructions"]
    actor_text = action_payload["messages"][0]["content"] if is_chat else action_payload["instructions"]
    common, separator, planning = planner_text.partition("Act as System 2: ")
    actor_common, actor_separator, acting = actor_text.partition("Act as System 1: ")
    if not separator or not actor_separator or common != actor_common:
        raise ValueError("Production role prompts changed; update the merged benchmark composition")
    instructions = (
        common
        + "In this single call, first choose the next plan, then produce the action chunk implementing that plan. "
        "Return both under the plan and action fields. The context plan is the previous plan; the current "
        "System 2 subgoal for your action is the new plan you generate in this reply. Planning requirements: "
        + planning
        + " Action requirements: "
        + acting
    )
    schema = _object_schema({"plan": PLAN_SCHEMA, "action": _action_schema(config.action_chunk_size)})
    effort = variant.removeprefix("merged_")
    # Match the two-stage sum of output/reasoning budgets instead of truncating
    # the merged condition to the smaller action-only budget.
    token_budget = (
        payload["max_completion_tokens" if is_chat else "max_output_tokens"]
        + action_payload["max_completion_tokens" if is_chat else "max_output_tokens"]
    )
    if is_chat:
        payload["messages"][0]["content"] = instructions
        payload["reasoning_effort"] = effort
        payload["max_completion_tokens"] = token_budget
        payload["response_format"]["json_schema"].update(name="merged_plan_action", schema=schema)
    else:
        payload.update(instructions=instructions, reasoning={"effort": effort}, max_output_tokens=token_budget)
        payload["text"]["format"].update(name="merged_plan_action", schema=schema)
    return urllib.request.Request(
        planner.full_url, json.dumps(payload, allow_nan=False).encode(), dict(planner.headers)
    )


def decode_result(document, config, role):
    """Enforce the production completion/refusal contract and validate every output waypoint."""
    if document.get("error"):
        raise ValueError("Provider error")
    if config.api_format == "chat_completions":
        choices = document.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Incomplete response")
        message = choices[0].get("message", {})
        if message.get("refusal") is not None:
            raise ValueError("Refused response")
        output = message.get("content")
    else:
        if document.get("status") != "completed":
            raise ValueError("Incomplete response")
        content = [item for entry in document.get("output", []) for item in entry.get("content", [])]
        if any(item.get("type") == "refusal" for item in content):
            raise ValueError("Refused response")
        output = "".join(item.get("text", "") for item in content if item.get("type") == "output_text")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Missing output")
    started = time.perf_counter()
    result = json.loads(output)
    parse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if role == "merged":
        _validate(result, _object_schema({"plan": PLAN_SCHEMA, "action": _action_schema(config.action_chunk_size)}))
        _validate_result("system2", result["plan"], config.action_chunk_size)
        _validate_result("system1", result["action"], config.action_chunk_size)
    else:
        _validate_result(role, result, config.action_chunk_size)
    return result, {
        "output_chars": len(output),
        "output_bytes": len(output.encode()),
        "output_parse_seconds": parse_seconds,
        "schema_validation_seconds": time.perf_counter() - started,
    }


def measure_call(config, variant, role, fixture, images, plan):
    """Measure a nonstreaming call; first body byte is not model first-token latency."""
    record = {"valid": False, "started_unix_seconds": time.time()}
    started = time.perf_counter()
    try:
        request = build_request(config, variant, role, fixture, images, plan)
        record["request_build_seconds"] = time.perf_counter() - started
        record["request_bytes"] = len(request.data)
        payload = json.loads(request.data)
        record["max_output_tokens"] = payload.get("max_completion_tokens", payload.get("max_output_tokens"))
        network_started = time.perf_counter()
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=config.system2_timeout) as response:
            record["headers_seconds"] = time.perf_counter() - network_started
            record["http_status"] = response.status
            record["response_headers"] = {
                key: value
                for key in ("x-request-id", "server-timing", "openai-processing-ms", "x-envoy-upstream-service-time")
                if (value := response.headers.get(key)) is not None
            }
            first = response.read(1)
            record["first_body_byte_seconds"] = time.perf_counter() - network_started
            body = first + response.read()
            record["full_response_seconds"] = time.perf_counter() - network_started
        record["response_bytes"] = len(body)
        parsed_at = time.perf_counter()
        document = json.loads(body)
        record["response_json_parse_seconds"] = time.perf_counter() - parsed_at
        record["usage"] = document.get("usage")  # Retains cache/reasoning detail when the provider reports it.
        record["provider_model"] = document.get("model")
        record["finish_reason"] = (
            [choice.get("finish_reason") for choice in document.get("choices", [])]
            if config.api_format == "chat_completions"
            else document.get("status")
        )
        result, details = decode_result(document, config, role)
        record.update(details, valid=True, result=result)
    except Exception as error:
        # No exception messages, response bodies, credential-bearing headers, or request dumps.
        record["error_type"] = type(error).__name__
        if isinstance(error, urllib.error.HTTPError):
            record["http_status"] = error.code
            error.close()
    record["total_seconds"] = time.perf_counter() - started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed calls from this exact experiment.")
    parser.add_argument(
        "--dry_run", action="store_true", help="Build all request variants without reading a key or HTTP."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("two_stage", "merged_high", "merged_medium"),
        default=["two_stage", "merged_high", "merged_medium"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max_calls", type=int, default=12, help="Total call cap, including calls reused with --resume."
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--goal", help="Override the fixture goal for every condition.")
    parser.add_argument("--api_key_file", type=Path)
    parser.add_argument("--api_key_env", default="NV_API_KEY")
    parser.add_argument("--base_url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--model", default="openai/openai/gpt-6-astra")
    parser.add_argument("--api_format", choices=("responses", "chat_completions"), default="chat_completions")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or not 1 <= args.max_calls <= 100:
        parser.error("repeats and max_calls must be between 1 and 100")
    if len(args.variants) != len(set(args.variants)):
        parser.error("variants must be unique")
    fixture = json.loads(args.fixture.read_text())
    images = []
    for view in fixture["image_paths"]:
        path = Path(view["path"])
        if not path.is_absolute():
            path = args.fixture.parent / path
        images.append(
            {"name": view["name"], "data_url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}
        )
    if sorted(view["name"] for view in images) != ["left_wrist_cam", "right_wrist_cam", "robot_pov_cam"]:
        parser.error("fixture must include the matching head, left-wrist, and right-wrist JPEGs")
    goal = args.goal or fixture.get("goal")
    if not goal:
        parser.error("fixture goal or --goal is required")
    key = "dry-run-placeholder" if args.dry_run else os.environ.get(args.api_key_env, "").strip()
    if not args.dry_run and args.api_key_file:
        key = args.api_key_file.expanduser().read_text().strip()
    if not key:
        parser.error("Set the API-key environment variable or --api_key_file")
    config = AstraConfig(
        goal=goal,
        api_key=key,
        model=args.model,
        base_url=args.base_url,
        api_format=args.api_format,
        action_chunk_size=8,
        waypoint_seconds=1.0,
        system1_timeout=args.timeout,
        system2_timeout=args.timeout,
    )
    fixture["action_spec"].update(action_chunk_size=8, waypoint_seconds=1.0)
    identity = {
        "fixture": fixture,
        "image_hashes": [hashlib.sha256(view["data_url"].encode()).hexdigest() for view in images],
        "config": {key: value for key, value in vars(config).items() if key != "api_key"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(
            Path(__file__).with_name("astra_teleop").joinpath("core.py").read_bytes()
        ).hexdigest(),
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        for variant in args.variants:
            for role in ["system2", "system1"] if variant == "two_stage" else ["merged"]:
                request = build_request(config, variant, role, fixture, images, fixture.get("plan"))
                print(
                    json.dumps(
                        {"variant": variant, "role": role, "request_bytes": len(request.data), "http_sent": False}
                    )
                )
        return
    previous = []
    if args.output.exists():
        if not args.resume:
            parser.error("output already exists; choose another file or --resume")
        previous = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        if any(row["signature"] != signature for row in previous):
            parser.error("resume fixture, model, prompts, or configuration differ from this output")
    completed = {(row["trial"], row["variant"], row["role"]): row for row in previous if row["kind"] == "call"}
    cycles = {(row["trial"], row["variant"]) for row in previous if row["kind"] == "cycle"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:

        def write(row):
            output.write(json.dumps({"signature": signature, **row}, allow_nan=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"result", "usage", "response_headers", "identity"}
                    }
                ),
                flush=True,
            )

        if not previous:
            write(
                {
                    "kind": "experiment",
                    "fixture_path": str(args.fixture.resolve()),
                    "identity": identity,
                    "clock_note": (
                        "Nonstreaming client timings; first body byte is not token TTFT. No physics execution."
                    ),
                }
            )
        for trial in range(args.repeats):
            rotation = trial % len(args.variants)
            for variant in args.variants[rotation:] + args.variants[:rotation]:
                if (trial, variant) in cycles:
                    continue
                records = []
                plan = fixture.get("plan")
                for role in ["system2", "system1"] if variant == "two_stage" else ["merged"]:
                    key = trial, variant, role
                    if key not in completed:
                        if len(completed) >= args.max_calls:
                            print(json.dumps({"stopped": "max_calls", "calls": len(completed)}), flush=True)
                            return
                        completed[key] = {
                            "kind": "call",
                            "trial": trial,
                            "variant": variant,
                            "role": role,
                            **measure_call(config, variant, role, fixture, images, plan),
                        }
                        write(completed[key])
                    record = completed[key]
                    records.append(record)
                    if not record["valid"]:
                        break
                    if role == "system2":
                        plan = record["result"]
                        if plan["stop"]:
                            break  # Production does not submit System 1 under a stopped plan.
                write(
                    {
                        "kind": "cycle",
                        "trial": trial,
                        "variant": variant,
                        "valid": all(row["valid"] for row in records),
                        "calls": len(records),
                        "action_produced": records[-1]["role"] != "system2" and records[-1]["valid"],
                        "total_call_seconds": sum(row["total_seconds"] for row in records),
                        "network_seconds": sum(row.get("full_response_seconds", 0) for row in records),
                    }
                )


if __name__ == "__main__":
    main()
