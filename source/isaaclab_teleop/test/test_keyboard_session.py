# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for keyboard teleop sessions fed by a scripted input surface in place of a visualizer window.

The ``se3_keyboard_teleop_cfg`` pipeline runs in a real Isaac Capture session. The in-process
keyboard never touches the OpenXR session handles, so a placeholder OpenXR session stands in for a
CloudXR runtime. The session lifecycle is checked without stepping, since its combined pipeline
also polls the controllers through OpenXR.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.keyboard import se3_keyboard_teleop_cfg
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
from isaacteleop.retargeting_engine.deviceio_source_nodes import FakeKeyEventSource, KeyboardSource, find_sources

_SETTLE_STEPS = 3


def _keyboard_cfg(**kwargs):
    """The SE(3) keyboard config, retargeting synchronously so each step reflects the current keys."""
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    cfg = se3_keyboard_teleop_cfg(**kwargs)
    cfg.retargeting_execution = RetargetingExecutionConfig(mode="sync")
    return cfg


class _PlaceholderOpenXRSession:
    """Non-null handles that only OpenXR-backed tracker impls would dereference."""

    def __init__(self, *args, **kwargs):
        pass

    def get_handles(self):
        from isaacteleop.oxr import OpenXRSessionHandles

        return OpenXRSessionHandles(1, 1, 1, 1)

    def get_provider_snapshot(self):
        return SimpleNamespace(state="AVAILABLE", headset_state="CONNECTED", reason="NONE", result_code=None, error="")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def surface():
    return FakeKeyEventSource()


@pytest.fixture
def keyboard_session(surface, monkeypatch):
    """The keyboard config's pipeline in a real Isaac Capture session, fed by *surface*."""
    from isaacteleop import oxr
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

    monkeypatch.setattr(oxr, "OpenXRSession", _PlaceholderOpenXRSession)
    cfg = _keyboard_cfg(pos_sensitivity=0.5, rot_sensitivity=0.5)
    pipeline = cfg.pipeline_builder()
    attachments = [keyboard.attach(surface) for keyboard in find_sources(pipeline, KeyboardSource)]
    session_config = TeleopSessionConfig(
        app_name="KeyboardSessionTest", pipeline=pipeline, retargeting_execution=cfg.retargeting_execution
    )
    with TeleopSession(session_config) as session:
        yield session
    for attachment in attachments:
        attachment.close()


def _settle(session):
    result = None
    for _ in range(_SETTLE_STEPS):
        result = session.step()
    return result["action"][0]


def test_held_key_drives_the_action(keyboard_session, surface):
    surface.press("KeyW")
    action = _settle(keyboard_session)
    assert action[0] > 0.0

    surface.release("KeyW")
    action = _settle(keyboard_session)
    assert action[0] == 0.0


def test_focus_loss_releases_held_keys(keyboard_session, surface):
    surface.press("KeyW")
    _settle(keyboard_session)

    surface.blur()
    action = _settle(keyboard_session)

    assert action[0] == 0.0


def _start(lifecycle):
    """Start a lifecycle without launching CloudXR, measuring the workstation or opening OpenXR."""
    import isaacteleop.teleop_session_manager as tsm

    with (
        patch.object(lifecycle, "_ensure_cloudxr_runtime") as ensure_cloudxr,
        patch.object(lifecycle, "_run_system_check"),
        patch.object(tsm, "TeleopSession", MagicMock()),
    ):
        lifecycle.start()
    return ensure_cloudxr


def _started_lifecycle(cfg, surface, **kwargs):
    lifecycle = TeleopSessionLifecycle(cfg, use_kit_xr_bridge=False, key_event_sources=[surface], **kwargs)
    return lifecycle, _start(lifecycle)


def test_keyboard_session_takes_the_standard_path(surface):
    """A keyboard session launches the CloudXR profile and uses the control channel, like any other."""
    lifecycle, ensure_cloudxr = _started_lifecycle(
        _keyboard_cfg(), surface, cloudxr_env_file="/nonexistent/cloudxr.env"
    )
    try:
        ensure_cloudxr.assert_called_once()
        assert lifecycle._message_processor is not None
        outputs = lifecycle._pipeline.output_types()
        assert TeleopSessionLifecycle._CONTROLLER_RIGHT_KEY in outputs
    finally:
        lifecycle.stop()


def test_surface_feeds_the_keyboard_and_the_control_keys(surface):
    """The pipeline's keyboard captures the surface's key bindings; the control-key listener only listens."""
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), surface)
    assert surface.listener_count == 2
    assert surface.capture_count == 1

    lifecycle.stop()
    assert surface.listener_count == 0
    assert not surface.captured


KEY_R, KEY_B = 19, 48


def test_each_control_key_press_is_drained_once(surface):
    """Presses reach the app once each, with no session step in between: repeats of a held key are dropped."""
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), surface)
    try:
        surface.press("KeyR")
        surface.press("KeyR")  # still held: a repeat, not a press
        surface.release("KeyR")
        surface.tap("KeyB")
        assert lifecycle.drain_pressed_keys() == [KEY_R, KEY_B]
        assert lifecycle.drain_pressed_keys() == []
    finally:
        lifecycle.stop()


def test_control_keys_work_without_a_pipeline_keyboard(surface):
    """A pipeline with no keyboard still gets B/P/R, and nothing captures the surface's bindings."""
    from isaaclab_teleop import se3_spacemouse_teleop_cfg

    lifecycle, _ = _started_lifecycle(se3_spacemouse_teleop_cfg(), surface)
    try:
        assert not surface.captured
        surface.tap("KeyB")
        assert lifecycle.drain_pressed_keys() == [KEY_B]
    finally:
        lifecycle.stop()


def test_presses_do_not_carry_into_the_next_session(surface):
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), surface)
    surface.tap("KeyR")
    lifecycle.stop()
    _start(lifecycle)
    try:
        surface.press("KeyR")  # the stale tap is gone; a new press counts
        assert lifecycle.drain_pressed_keys() == [KEY_R]
    finally:
        lifecycle.stop()


def test_another_surface_losing_focus_does_not_repeat_a_held_key():
    """Each surface tracks its own held keys, so one surface's blur leaves another's repeat suppression intact."""
    first, second = FakeKeyEventSource(), FakeKeyEventSource()
    lifecycle = TeleopSessionLifecycle(_keyboard_cfg(), use_kit_xr_bridge=False, key_event_sources=[first, second])
    _start(lifecycle)
    try:
        first.press("KeyR")
        second.blur()
        first.press("KeyR")  # an autorepeat of the key the first surface still holds
        assert lifecycle.drain_pressed_keys() == [KEY_R]
    finally:
        lifecycle.stop()


def test_a_late_callback_from_a_stopped_session_is_ignored(surface):
    """A key callback still running when the session stops must not queue into the next session."""
    key_callbacks = []
    add_key_listener = surface.add_key_listener

    def recording_add_key_listener(on_key, on_focus_lost):
        key_callbacks.append(on_key)
        return add_key_listener(on_key, on_focus_lost)

    surface.add_key_listener = recording_add_key_listener
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), surface)
    # The control-key listener (attached after the pipeline's keyboard), as a surface's event
    # thread would hold its callback mid-call.
    late_on_key = key_callbacks[-1]
    lifecycle.stop()
    _start(lifecycle)
    try:
        late_on_key("KeyR", True)  # the held callback resumes after the restart
        assert lifecycle.drain_pressed_keys() == []
    finally:
        lifecycle.stop()
