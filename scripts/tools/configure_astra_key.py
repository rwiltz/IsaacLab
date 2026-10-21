#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prompt locally for your inference provider's API key and save it for Astra teleoperation."""

import argparse
import getpass
import os
import tempfile
import warnings
from pathlib import Path


def main() -> None:
    """Replace the key file atomically, without displaying the key or accepting it on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, default=Path.home() / ".config/openai/api_key", help="Destination key file."
    )
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("New inference-provider API key (hidden): ").strip()
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        raise SystemExit("Key unchanged. Run this script in an interactive host terminal.") from None
    if not key or any(character.isspace() for character in key):
        raise SystemExit("Key unchanged. Enter a nonempty key without whitespace.")

    destination = args.path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, prefix=".api_key-", delete=False) as file:
            temporary = Path(file.name)
            os.fchmod(file.fileno(), 0o600)
            file.write(key + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Key saved to {destination} with owner-only permissions (0600).")
    print("Use the matching provider's --base_url when launching Astra.")


if __name__ == "__main__":
    main()
