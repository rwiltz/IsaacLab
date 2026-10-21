# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise inference transport and stale-action boundaries without launching a simulator."""

import copy
import io
import json
import queue
import sys
import time
import urllib.error
import urllib.response
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.core import ActionContract, AstraConfig, AstraHTTPClient, AstraPipeline, MockClient  # noqa: E402

PLAN = {"subgoal": "Move the right hand toward the cube", "success_criteria": "Hand is next to cube", "stop": False}
COMMAND = {
    "left_wrist": {"position": [0.3, 0.2, 0.9], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    "right_wrist": {"position": [0.3, -0.2, 0.9], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    "left_grip_closed": False,
    "right_grip_closed": False,
}
ACTION = {"command": COMMAND, "hold_seconds": 1.0, "stop": False}


def intercept_http(monkeypatch, respond):
    """Replace network I/O while preserving production request construction."""
    monkeypatch.setattr(
        "urllib.request.OpenerDirector.open", lambda self, request, **kwargs: respond(request, **kwargs)
    )


class ControlledClient:
    """Keep requests pending until the test supplies their external inference result."""

    def __init__(self):
        self.requests = queue.Queue()

    def infer(self, role, snapshot, images, action_spec, plan):
        result = queue.Queue()
        self.requests.put((role, snapshot, result))
        value = result.get(timeout=2)
        if isinstance(value, Exception):
            raise value
        return value


def receive(pipeline, now, event):
    """Poll the real worker boundary until the expected event arrives."""
    deadline = time.monotonic() + 2
    collected = []
    while time.monotonic() < deadline:
        command = pipeline.poll(now=now)
        collected.extend(pipeline.drain_events())
        if any(item["event"] == event for item in collected):
            return command, collected
        time.sleep(0.001)
    pytest.fail(f"No {event} event received: {collected}")


@pytest.fixture
def control():
    client = ControlledClient()
    pipeline = AstraPipeline(AstraConfig(goal="Reach the cube", system2_interval=1.0, system1_max_age=5.0), client)
    yield pipeline, client
    pipeline.close()


def start_plan(pipeline, client):
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.0)
    role, _, result = client.requests.get(timeout=1)
    assert role == "system2"
    result.put(copy.deepcopy(PLAN))
    command, _ = receive(pipeline, 100.1, "accepted")
    assert command is None


def test_action_uses_detached_snapshot_and_expires(control):
    pipeline, client = control
    start_plan(pipeline, client)
    snapshot = {"neutral_command": copy.deepcopy(COMMAND)}
    pipeline.poll(snapshot, [], {}, now=100.2)
    role, submitted, result = client.requests.get(timeout=1)
    snapshot["neutral_command"]["left_wrist"]["position"][2] = 0.0
    assert role == "system1"
    assert submitted["neutral_command"]["left_wrist"]["position"][2] == 0.9
    result.put(copy.deepcopy(ACTION))
    command, _ = receive(pipeline, 100.3, "accepted")
    assert command == COMMAND
    assert pipeline.status["command_id"] == 1
    command["left_wrist"]["position"][2] = 0.0
    assert pipeline.poll(now=100.4)["left_wrist"]["position"][2] == 0.9
    assert pipeline.poll(now=101.31) is None


def test_reset_discards_pending_work_without_accumulating_requests(control):
    pipeline, client = control
    pipeline.poll({}, [], {}, now=100.0)
    _, _, result = client.requests.get(timeout=1)
    for now in (100.1, 100.2, 100.3):
        pipeline.reset()
        assert pipeline.poll({}, [], {}, now=now) is None
    assert client.requests.empty()
    result.put(copy.deepcopy(PLAN))
    command, events = receive(pipeline, 100.4, "discarded")
    assert command is None
    assert pipeline.status["plan"] is None
    assert any(item.get("reason") == "outdated" for item in events)
    assert pipeline.needs_snapshot(now=100.5)


def test_new_plan_preempts_inflight_action(control):
    pipeline, client = control
    start_plan(pipeline, client)
    pipeline.poll({}, [], {}, now=100.2)
    role, _, old_action = client.requests.get(timeout=1)
    assert role == "system1"
    pipeline.poll({}, [], {}, now=101.1)
    role, _, new_plan = client.requests.get(timeout=1)
    assert role == "system2"
    new_plan.put({**PLAN, "subgoal": "Stop moving", "stop": True})
    assert receive(pipeline, 101.2, "accepted")[0] is None
    old_action.put(copy.deepcopy(ACTION))
    command, events = receive(pipeline, 101.3, "discarded")
    assert command is None
    assert pipeline.status["plan"]["stop"] is True
    assert any(event.get("reason") == "outdated" for event in events)


def test_plan_expiry_keeps_fresh_planner_but_discards_old_action():
    client = ControlledClient()
    pipeline = AstraPipeline(AstraConfig(goal="Reach", system1_max_age=10.0), client)
    try:
        pipeline.poll({}, [], {}, now=100.0)
        _, _, initial_plan = client.requests.get(timeout=1)
        initial_plan.put(copy.deepcopy(PLAN))
        receive(pipeline, 125.0, "accepted")
        pipeline.poll({}, [], {}, now=125.1)
        role, _, refreshed_plan = client.requests.get(timeout=1)
        assert role == "system2"
        role, _, old_action = client.requests.get(timeout=1)
        assert role == "system1"

        assert pipeline.poll(now=130.1) is None
        old_action.put(copy.deepcopy(ACTION))
        assert receive(pipeline, 130.2, "discarded")[0] is None
        # This refresh has its own unexpired snapshot, despite its predecessor's expiry.
        refreshed_plan.put(copy.deepcopy(PLAN))
        assert receive(pipeline, 150.0, "accepted")[0] is None
        assert pipeline.status["plan"] == PLAN
        assert pipeline.needs_snapshot(now=150.1)
    finally:
        pipeline.close()


def test_action_stop_discards_planner_started_before_stop(control):
    pipeline, client = control
    start_plan(pipeline, client)
    pipeline.poll({}, [], {}, now=101.1)
    role, _, old_plan = client.requests.get(timeout=1)
    assert role == "system2"
    role, _, action = client.requests.get(timeout=1)
    assert role == "system1"
    action.put({**copy.deepcopy(ACTION), "stop": True})
    assert receive(pipeline, 101.2, "accepted")[0] is None
    old_plan.put(copy.deepcopy(PLAN))
    assert receive(pipeline, 101.3, "discarded")[0] is None
    assert pipeline.status["plan"] is None
    assert pipeline.needs_snapshot(now=101.4)


@pytest.mark.parametrize("failure", ["stale", "timeout", "invalid"])
def test_unusable_inference_never_becomes_an_action(control, failure):
    pipeline, client = control
    start_plan(pipeline, client)
    pipeline.poll({}, [], {}, now=100.2)
    _, _, result = client.requests.get(timeout=1)
    if failure == "timeout":
        assert pipeline.poll(now=121.0) is None
        assert any(event.get("error") == "request_timeout" for event in pipeline.drain_events())
        result.put(copy.deepcopy(ACTION))
        command, _ = receive(pipeline, 121.1, "discarded")
    elif failure == "stale":
        result.put(copy.deepcopy(ACTION))
        command, events = receive(pipeline, 106.0, "discarded")
        assert any(event.get("reason") == "stale_snapshot" for event in events)
    else:
        result.put(RuntimeError("secret credential must never be logged"))
        command, events = receive(pipeline, 100.3, "error")
        assert "secret credential" not in json.dumps(events)
    assert command is None


@pytest.mark.parametrize("api_format", ["responses", "chat_completions"])
def test_http_transport_sends_images_and_strict_schema(monkeypatch, api_format):
    payloads = []

    def respond(request, timeout):
        endpoint = "/responses" if api_format == "responses" else "/chat/completions"
        assert request.full_url == "https://inference.example/v1" + endpoint
        payloads.append(json.loads(request.data))
        if api_format == "responses":
            response = {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(ACTION)}]}],
            }
        else:
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(ACTION), "refusal": None},
                    }
                ]
            }
        return io.BytesIO(json.dumps(response).encode())

    intercept_http(monkeypatch, respond)
    config = AstraConfig(
        goal="Reach", api_key="test-key", api_format=api_format, base_url="https://inference.example/v1"
    )
    client = AstraHTTPClient(config)
    image = {"name": "head", "data_url": "data:image/jpeg;base64,dGVzdA=="}
    assert client.infer("system1", {"step": 1}, [image], {}, PLAN) == ACTION
    payload = payloads[0]
    assert payload["model"] == "openai/openai/gpt-6-astra"
    assert payload["store"] is False
    assert payload["stream"] is False
    if api_format == "responses":
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["text"]["format"]["strict"] is True
        assert payload["input"][0]["content"][-1]["image_url"] == image["data_url"]
        assert payload["max_output_tokens"] == 4096
    else:
        assert payload["reasoning_effort"] == "low"
        assert payload["max_completion_tokens"] == 4096
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"][0]["type"] == "text"
        assert payload["messages"][1]["content"][-1] == {
            "type": "image_url",
            "image_url": {"url": image["data_url"], "detail": "low"},
        }
    assert "test-key" not in repr(config)


@pytest.mark.parametrize(
    ("api_format", "failure"),
    [
        ("responses", "incomplete"),
        ("responses", "refusal"),
        ("responses", "empty"),
        ("responses", "shape"),
        ("responses", "nan"),
        ("responses", "zero_quaternion"),
        ("responses", "bool"),
        ("responses", "grip_type"),
        ("responses", "legacy_hands"),
        ("chat_completions", "incomplete"),
        ("chat_completions", "refusal"),
        ("chat_completions", "empty"),
        ("chat_completions", "grip_type"),
    ],
)
def test_http_transport_rejects_unusable_output(monkeypatch, api_format, failure):
    action = copy.deepcopy(ACTION)
    if failure == "shape":
        action["command"]["left_wrist"]["position"] = [0.0]
    elif failure == "nan":
        action["command"]["left_wrist"]["position"][2] = float("nan")
    elif failure == "zero_quaternion":
        action["command"]["left_wrist"]["quaternion"] = [0.0] * 4
    elif failure == "bool":
        action["command"]["left_wrist"]["position"][2] = True
    elif failure == "grip_type":
        action["command"]["left_grip_closed"] = "closed"
    elif failure == "legacy_hands":
        action["command"].update(left_hand=[0.0] * 7, right_hand=[0.0] * 7)
    if api_format == "responses":
        content = [{"type": "output_text", "text": json.dumps(action)}]
        if failure == "refusal":
            content = [{"type": "refusal", "refusal": "Cannot comply"}]
        elif failure == "empty":
            content = []
        response = {
            "status": "incomplete" if failure == "incomplete" else "completed",
            "output": [{"type": "message", "content": content}],
        }
    else:
        response = {
            "choices": [
                {
                    "finish_reason": "length" if failure == "incomplete" else "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if failure == "empty" else json.dumps(action),
                        "refusal": "Cannot comply" if failure == "refusal" else None,
                    },
                }
            ]
        }
    intercept_http(monkeypatch, lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    client = AstraHTTPClient(AstraConfig(goal="Reach", api_key="test-key", api_format=api_format))
    with pytest.raises(ValueError):
        client.infer("system1", {}, [], {}, PLAN)


@pytest.fixture
def chunk_control():
    client = ControlledClient()
    pipeline = AstraPipeline(
        AstraConfig(
            goal="Reach",
            action_chunk_size=3,
            waypoint_seconds=0.5,
            system1_interval=0.1,
            system1_max_age=5.0,
            system2_interval=10.0,
            system2_max_age=100.0,
        ),
        client,
    )
    start_plan(pipeline, client)
    yield pipeline, client
    pipeline.close()


def motion_chunk(offset=0.0, feedback_after_waypoints=0):
    commands = [copy.deepcopy(COMMAND) for _ in range(3)]
    for index, command in enumerate(commands):
        command["right_wrist"]["position"][2] += offset + index * 0.01
    return {"commands": commands, "feedback_after_waypoints": feedback_after_waypoints, "stop": False}


def start_chunk(pipeline, client, feedback_after_waypoints=0):
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.2)
    role, _, result = client.requests.get(timeout=1)
    assert role == "system1"
    chunk = motion_chunk(feedback_after_waypoints=feedback_after_waypoints)
    result.put(chunk)
    assert receive(pipeline, 100.3, "accepted")[0] == chunk["commands"][0]
    return chunk


def queue_chunk(pipeline, client):
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.4)
    role, snapshot, result = client.requests.get(timeout=1)
    assert role == "system1"
    chunk = motion_chunk(0.1)
    result.put(chunk)
    receive(pipeline, 100.5, "chunk_queued")
    return chunk, snapshot


def test_chunks_prefetch_without_overwrite_and_handoff_on_simulation_time(chunk_control):
    pipeline, client = chunk_control
    first = start_chunk(pipeline, client)
    pipeline.poll({"neutral_command": COMMAND, "latest_observation": 42}, [], {}, now=100.4)
    role, snapshot, result = client.requests.get(timeout=1)
    assert role == "system1"
    assert snapshot["latest_observation"] == 42
    assert snapshot["action_queue"]["remaining_commands"] == first["commands"]
    assert snapshot["action_queue"]["expected_start_command"] == first["commands"][-1]
    assert snapshot["action_queue"]["expected_start_is_prediction"] is True
    assert snapshot["action_queue"]["remaining_sim_seconds"] == 1.5
    assert pipeline.status["chunk_id"] == snapshot["action_queue"]["predecessor_chunk_id"]
    assert pipeline.status["in_flight"] == ["system1"]
    pipeline.advance(0.49)
    second = motion_chunk(0.1)
    result.put(second)
    assert receive(pipeline, 100.5, "chunk_queued")[0] == first["commands"][0]
    assert pipeline.status["command_id"] == 1
    assert pipeline.status["queued_chunk_id"] == 2
    assert not pipeline.needs_snapshot(now=102.0)
    assert pipeline.poll({"neutral_command": COMMAND}, [], {}, now=102.0) == first["commands"][0]
    assert client.requests.empty()

    pipeline.advance(0.01)
    assert pipeline.poll(now=102.1) == first["commands"][1]
    assert pipeline.status["command_id"] == 2
    # A slow control tick cannot skip unexecuted waypoints or debit the next one.
    pipeline.advance(10.0)
    assert pipeline.poll(now=102.2) == first["commands"][2]
    pipeline.advance(0.5)
    assert pipeline.poll(now=102.3) == second["commands"][0]
    assert pipeline.status["command_id"] == 4
    assert pipeline.status["chunk_id"] == 2
    assert pipeline.status["remaining_sim_seconds"] == 1.5
    assert pipeline.status["queued_chunk_id"] is None
    assert not any(event["event"] == "underrun" for event in pipeline.drain_events())
    pipeline.advance(0.49)
    assert pipeline.poll(now=102.4) == second["commands"][0]
    pipeline.advance(0.01)
    assert pipeline.poll(now=102.5) == second["commands"][1]


@pytest.mark.parametrize("failure", ["error", "timeout", "stale", "bad_later_waypoint"])
def test_failed_prefetch_preserves_active_chunk_and_rejects_whole_bad_chunk(chunk_control, failure):
    pipeline, client = chunk_control
    first = start_chunk(pipeline, client)
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.4)
    _, _, result = client.requests.get(timeout=1)
    if failure == "timeout":
        assert pipeline.poll(now=121.0) == first["commands"][0]
        assert any(event.get("error") == "request_timeout" for event in pipeline.drain_events())
        result.put(motion_chunk(0.1))
        assert receive(pipeline, 121.1, "discarded")[0] == first["commands"][0]
    else:
        response = motion_chunk(0.1)
        if failure == "bad_later_waypoint":
            response["commands"][-1]["right_wrist"]["quaternion"] = [0.0] * 4
        result.put(RuntimeError("request failed") if failure == "error" else response)
        now, event = (106.0, "discarded") if failure == "stale" else (100.5, "error")
        assert receive(pipeline, now, event)[0] == first["commands"][0]
    assert pipeline.status["chunk_id"] == 1
    assert pipeline.status["queued_chunk_id"] is None
    assert pipeline.status["command_id"] == 1


def test_stale_queued_chunk_is_discarded_at_activation_and_underrun_is_reported_once(chunk_control):
    pipeline, client = chunk_control
    first = start_chunk(pipeline, client)
    queue_chunk(pipeline, client)
    for index in range(3):
        # Active waypoints do not expire solely because their original snapshot ages.
        assert pipeline.poll(now=106.0) == first["commands"][index]
        pipeline.advance(0.5)
    assert pipeline.poll(now=106.0) is None
    events = pipeline.drain_events()
    assert sum(event["event"] == "underrun" for event in events) == 1
    assert any(event.get("reason") == "stale_snapshot" and event.get("chunk_id") == 2 for event in events)
    assert pipeline.status["queued_chunk_id"] is None
    assert pipeline.poll(now=106.1) is None
    assert not any(event["event"] == "underrun" for event in pipeline.drain_events())
    assert pipeline.needs_snapshot(now=106.1)


def test_reset_clears_active_queued_and_inflight_planner(chunk_control):
    pipeline, client = chunk_control
    start_chunk(pipeline, client)
    queue_chunk(pipeline, client)
    pipeline.poll({}, [], {}, now=110.1)
    role, _, result = client.requests.get(timeout=1)
    assert role == "system2"
    pipeline.reset()
    assert pipeline.poll(now=110.2) is None
    assert pipeline.status["chunk_id"] is None and pipeline.status["queued_chunk_id"] is None
    result.put(copy.deepcopy(PLAN))
    assert receive(pipeline, 110.3, "discarded")[0] is None


def test_new_plan_preempts_active_chunk_and_inflight_prefetch(chunk_control):
    pipeline, client = chunk_control
    start_chunk(pipeline, client)
    pipeline.poll({}, [], {}, now=100.4)
    _, _, old_chunk = client.requests.get(timeout=1)
    pipeline.poll({}, [], {}, now=110.1)
    role, _, new_plan = client.requests.get(timeout=1)
    assert role == "system2"
    new_plan.put({**PLAN, "subgoal": "Move the left hand toward the cube"})
    assert receive(pipeline, 110.2, "accepted")[0] is None
    old_chunk.put(motion_chunk(0.1))
    assert receive(pipeline, 110.3, "discarded")[0] is None
    assert pipeline.status["chunk_id"] is None and pipeline.status["queued_chunk_id"] is None


def test_planner_stop_clears_both_chunks(chunk_control):
    pipeline, client = chunk_control
    start_chunk(pipeline, client)
    queue_chunk(pipeline, client)
    pipeline.poll({}, [], {}, now=110.1)
    _, _, result = client.requests.get(timeout=1)
    result.put({**PLAN, "stop": True})
    assert receive(pipeline, 110.2, "accepted")[0] is None
    assert pipeline.status["chunk_id"] is None and pipeline.status["queued_chunk_id"] is None


def test_action_stop_preempts_active_chunk_and_older_planner(chunk_control):
    pipeline, client = chunk_control
    start_chunk(pipeline, client)
    pipeline.poll({}, [], {}, now=110.1)
    role, _, old_plan = client.requests.get(timeout=1)
    assert role == "system2"
    role, _, result = client.requests.get(timeout=1)
    assert role == "system1"
    result.put({**motion_chunk(0.1), "stop": True})
    assert receive(pipeline, 110.2, "accepted")[0] is None
    old_plan.put(copy.deepcopy(PLAN))
    assert receive(pipeline, 110.3, "discarded")[0] is None
    assert pipeline.status["chunk_id"] is None and pipeline.status["queued_chunk_id"] is None


def test_feedback_checkpoint_requires_post_execution_planner_and_discards_tail(chunk_control):
    pipeline, client = chunk_control
    chunk = start_chunk(pipeline, client, feedback_after_waypoints=2)
    assert pipeline.status["feedback_after_waypoints"] == 2
    assert pipeline.status["remaining_sim_seconds"] == 1.0
    assert not pipeline.needs_snapshot(now=100.4)
    pipeline.poll({}, [], {}, now=110.1)
    role, before, old_plan = client.requests.get(timeout=1)
    assert role == "system2"
    assert before["action_queue"]["remaining_commands"] == chunk["commands"][:2]
    assert before["action_queue"]["expected_start_command"] == chunk["commands"][1]
    assert before["action_queue"]["remaining_sim_seconds"] == 1.0
    assert client.requests.empty()  # A checkpointed chunk cannot prefetch its successor.
    pipeline.advance(0.49)
    assert pipeline.poll(now=110.2) == chunk["commands"][0]
    assert not pipeline.status["awaiting_feedback"]
    pipeline.advance(0.01)
    assert pipeline.poll(now=110.3) == chunk["commands"][1]
    pipeline.advance(0.5)
    assert pipeline.poll(now=110.4) is None
    assert pipeline.status["awaiting_feedback"]
    assert pipeline.status["plan"] == PLAN  # The runner continues holding the wrist/grip targets.
    assert pipeline.status["chunk_id"] is None
    assert pipeline.status["command_id"] == 2
    events = pipeline.drain_events()
    assert sum(event["event"] == "feedback_checkpoint" for event in events) == 1
    assert not any(event["event"] == "underrun" for event in events)
    assert not pipeline.needs_snapshot(now=110.4)  # The bounded older worker must drain first.

    old_plan.put(copy.deepcopy(PLAN))
    assert receive(pipeline, 110.5, "discarded")[0] is None
    assert pipeline.status["awaiting_feedback"]
    assert pipeline.needs_snapshot(now=110.6)  # Bypass the previous planner's 120.1-second due time.
    pipeline.poll({"neutral_command": COMMAND, "step": 52}, [], {}, now=110.6)
    role, after, fresh_plan = client.requests.get(timeout=1)
    assert role == "system2"
    assert after["step"] == 52
    assert after["feedback_checkpoint"] == {"chunk_id": 1, "completed_waypoints": 2}
    assert after["action_queue"]["remaining_commands"] == []
    assert after["action_queue"]["expected_start_command"] == COMMAND
    assert after["action_queue"]["expected_start_is_prediction"] is False
    assert client.requests.empty()
    fresh_plan.put(copy.deepcopy(PLAN))
    assert receive(pipeline, 110.7, "feedback_ready")[0] is None
    assert not pipeline.status["awaiting_feedback"]
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=110.8)
    role, _, next_action = client.requests.get(timeout=1)
    assert role == "system1"
    next_action.put(motion_chunk(0.1))
    assert receive(pipeline, 110.9, "chunk_started")[0] == motion_chunk(0.1)["commands"][0]
    assert pipeline.status["command_id"] == 3  # The discarded third waypoint never became active.


@pytest.mark.parametrize("interrupt", ["reset", "stop"])
def test_feedback_wait_is_cleared_by_reset_or_planner_stop(chunk_control, interrupt):
    pipeline, client = chunk_control
    count = 1 if interrupt == "reset" else 3
    chunk = start_chunk(pipeline, client, feedback_after_waypoints=count)
    for index in range(count):
        assert pipeline.poll(now=100.35) == chunk["commands"][index]
        pipeline.advance(0.5)
    pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.4)
    role, _, result = client.requests.get(timeout=1)
    assert role == "system2"
    assert pipeline.status["awaiting_feedback"]
    if interrupt == "reset":
        pipeline.reset()
        result.put(copy.deepcopy(PLAN))
        assert receive(pipeline, 100.5, "discarded")[0] is None
        assert pipeline.status["plan"] is None
    else:
        result.put({**PLAN, "stop": True})
        assert receive(pipeline, 100.5, "feedback_ready")[0] is None
        assert pipeline.status["plan"]["stop"]
    assert not pipeline.status["awaiting_feedback"]
    assert pipeline.status["chunk_id"] is None and pipeline.status["queued_chunk_id"] is None


def test_http_chunk_schema_and_mock_contract(monkeypatch):
    chunk = motion_chunk(feedback_after_waypoints=3)

    def respond(request, timeout):
        payload = json.loads(request.data)
        schema = payload["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == {"commands", "feedback_after_waypoints", "stop"}
        assert schema["properties"]["commands"]["minItems"] == 3
        assert schema["properties"]["commands"]["maxItems"] == 3
        assert schema["properties"]["feedback_after_waypoints"] == {"type": "integer", "minimum": 0, "maximum": 3}
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(chunk)},
                        }
                    ]
                }
            ).encode()
        )

    intercept_http(monkeypatch, respond)
    client = AstraHTTPClient(AstraConfig(goal="Reach", api_key="test", action_chunk_size=3))
    assert client.infer("system1", {}, [], {}, PLAN) == chunk
    pipeline = AstraPipeline(AstraConfig(goal="Hold", action_chunk_size=3), MockClient())
    try:
        pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.0)
        receive(pipeline, 100.1, "accepted")
        pipeline.poll({"neutral_command": COMMAND}, [], {}, now=100.2)
        assert receive(pipeline, 100.3, "accepted")[0] == COMMAND
        assert pipeline.status["remaining_sim_seconds"] == 1.5
    finally:
        pipeline.close()


@pytest.mark.parametrize("count", [True, 1.0, -1, 4])
def test_http_chunk_rejects_invalid_feedback_count(monkeypatch, count):
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(motion_chunk(feedback_after_waypoints=count))},
            }
        ]
    }
    intercept_http(monkeypatch, lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    client = AstraHTTPClient(AstraConfig(goal="Reach", api_key="test", action_chunk_size=3))
    with pytest.raises(ValueError):
        client.infer("system1", {}, [], {}, PLAN)


@pytest.fixture
def force_contract():
    def validate_mode(command):
        if command["mode"] == "idle" and any(command["forces"]):
            raise ValueError("Idle mode requires zero force")

    return ActionContract(
        command_schema={
            "type": "object",
            "properties": {
                "forces": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -1, "maximum": 1},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "mode": {"type": "string", "enum": ["idle", "drive"]},
            },
            "required": ["forces", "mode"],
            "additionalProperties": False,
        },
        instructions="Apply normalized actuator forces; idle mode must command zero force.",
        command_validator=validate_mode,
    )


@pytest.mark.parametrize(("api_format", "chunk_size"), [("responses", 1), ("chat_completions", 3)])
def test_custom_http_contract_uses_detached_schema_and_environment_instructions(
    monkeypatch, force_contract, api_format, chunk_size
):
    source_schema = force_contract.command_schema
    contract = ActionContract(source_schema, force_contract.instructions)
    source_schema["properties"]["forces"]["items"]["maximum"] = 100
    contract.command_schema["properties"]["forces"]["items"]["maximum"] = 100
    command = {"forces": [0.25, -0.5], "mode": "drive"}
    result = (
        {"command": command, "hold_seconds": 0.5, "stop": False}
        if chunk_size == 1
        else {"commands": [command] * chunk_size, "feedback_after_waypoints": 0, "stop": False}
    )

    def respond(request, timeout):
        payload = json.loads(request.data)
        if api_format == "responses":
            schema = payload["text"]["format"]["schema"]
            instructions = payload["instructions"]
            response = {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(result)}]}],
            }
        else:
            schema = payload["response_format"]["json_schema"]["schema"]
            instructions = payload["messages"][0]["content"]
            response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
        command_schema = (
            schema["properties"]["command"] if chunk_size == 1 else schema["properties"]["commands"]["items"]
        )
        assert command_schema == force_contract.command_schema
        assert force_contract.instructions in instructions
        return io.BytesIO(json.dumps(response).encode())

    intercept_http(monkeypatch, respond)
    client = AstraHTTPClient(
        AstraConfig(goal="Balance", api_key="test", api_format=api_format, action_chunk_size=chunk_size),
        action_contract=contract,
    )
    assert client.infer("system1", {}, [], {}, PLAN) == result
    command["forces"][0] = 2.0
    with pytest.raises(ValueError, match="custom action schema"):
        client.infer("system1", {}, [], {}, PLAN)


@pytest.mark.parametrize(
    "command",
    [
        {"forces": [True], "mode": "drive"},
        {"forces": [float("nan")], "mode": "drive"},
        {"forces": [float("inf")], "mode": "drive"},
        {"forces": [0, 0, 0], "mode": "drive"},
        {"forces": [0], "mode": "unknown"},
        {"forces": [0], "mode": "drive", "unexpected": 0},
        {"forces": [0.5], "mode": "idle"},
    ],
)
def test_custom_contract_enforces_schema_finite_values_and_semantics(force_contract, command):
    with pytest.raises(ValueError):
        force_contract.validate_command(command)


def test_custom_pipeline_rejects_entire_chunk_from_non_http_client(force_contract):
    client = ControlledClient()
    pipeline = AstraPipeline(AstraConfig(goal="Balance", action_chunk_size=3), client, action_contract=force_contract)
    try:
        start_plan(pipeline, client)
        pipeline.poll({}, [], {}, now=100.2)
        _, _, result = client.requests.get(timeout=1)
        result.put(
            {
                "commands": [
                    {"forces": [0], "mode": "idle"},
                    {"forces": [0.25], "mode": "drive"},
                    {"forces": [0.25], "mode": "idle"},
                ],
                "feedback_after_waypoints": 0,
                "stop": False,
            }
        )
        command, events = receive(pipeline, 100.3, "error")
        assert command is None and pipeline.status["chunk_id"] is None
        assert any(event.get("error") == "ValueError" for event in events)
    finally:
        pipeline.close()


def test_custom_mock_runs_adapter_neutral_command_through_chunk_scheduler(force_contract):
    config = AstraConfig(goal="Balance", action_chunk_size=3)
    client = MockClient(action_contract=force_contract)
    with pytest.raises(ValueError, match="same action_contract"):
        AstraPipeline(config, client)
    pipeline = AstraPipeline(config, client, action_contract=force_contract)
    neutral = {"forces": [0], "mode": "idle"}
    try:
        pipeline.poll({"neutral_command": neutral}, [], {}, now=100.0)
        receive(pipeline, 100.1, "accepted")
        pipeline.poll({"neutral_command": neutral}, [], {}, now=100.2)
        assert receive(pipeline, 100.3, "accepted")[0] == neutral
        pipeline.advance(0.5)
        assert pipeline.poll(now=100.4) == neutral
        assert pipeline.status["command_id"] == 2
        pipeline.reset()
        assert pipeline.poll(now=100.5) is None
    finally:
        pipeline.close()


def test_custom_contract_rejects_references_before_any_request():
    with pytest.raises(ValueError, match="inline"):
        ActionContract(
            {"type": "object", "properties": {"force": {"$ref": "https://example.com/force.json"}}},
            "Apply a force.",
        )


def test_http_transport_rejects_redirect_without_sending_credentials_to_second_endpoint(monkeypatch):
    attempted_urls = []

    def respond(handler, request):
        attempted_urls.append(request.full_url)
        headers = Message()
        headers["Content-Type"] = "application/json"
        if len(attempted_urls) == 1:
            headers["Location"] = "https://other.example/collect"
            response = urllib.response.addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
            response.msg = "Found"
        else:
            document = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ACTION)}}]}
            response = urllib.response.addinfourl(
                io.BytesIO(json.dumps(document).encode()), headers, request.full_url, 200
            )
            response.msg = "OK"
        return response

    # Exercise urllib's real redirect/error processing, replacing only socket I/O.
    monkeypatch.setattr("urllib.request.HTTPSHandler.https_open", respond)
    client = AstraHTTPClient(
        AstraConfig(goal="Hold", api_key="synthetic-test-key", base_url="https://inference.example/v1")
    )
    with pytest.raises(urllib.error.HTTPError) as failure:
        client.infer("system1", {}, [], {}, PLAN)
    assert failure.value.code == 302
    failure.value.close()
    assert attempted_urls == ["https://inference.example/v1/chat/completions"]
