# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line handling shared by the Astra launchers."""

import logging
import sys


def normalize_cli_args(argv: list[str] | None) -> list[str]:
    """Remove rich-text padding that shells preserve at argument boundaries.

    Slack and document editors can replace indentation with non-breaking spaces.
    Trim those characters at token edges without splitting or rewriting quoted
    goals, paths, or other argument contents.
    """
    original = list(sys.argv[1:] if argv is None else argv)
    normalized = [
        argument.strip("\u00a0\u2007\u202f")
        for argument in original
        if argument == "" or argument.strip("\u00a0\u2007\u202f")
    ]
    if normalized != original:
        logging.getLogger(__name__).warning("Removed non-breaking space padding from copied command-line arguments.")
    return normalized
