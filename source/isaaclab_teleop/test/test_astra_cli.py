# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect pasted launcher arguments without changing quoted argument contents."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.cli import normalize_cli_args  # noqa: E402


def test_rich_text_padding_parses_flags_values_and_preserves_goal_and_empty_selection():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_drop", action="store_true")
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--goal")
    parser.add_argument("--cameras")
    goal = "Hold the wheel\u00a0above the bin; keep  internal spaces."
    pasted = [
        "\u00a0",
        "\u00a0--bin_drop",
        "\u2007--max_episodes",
        "\u00a02",
        "\u202f--max_steps",
        "120\u202f",
        "--goal",
        goal,
        "--cameras",
        "",
        "\u00a0renderer=none\u2007",
    ]

    args, overrides = parser.parse_known_args(normalize_cli_args(pasted))

    assert args.bin_drop is True
    assert args.max_episodes == 2
    assert args.max_steps == 120
    assert args.goal == goal
    assert args.cameras == ""
    assert overrides == ["renderer=none"]
