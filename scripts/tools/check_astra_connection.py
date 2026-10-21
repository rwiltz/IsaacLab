#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify an NVIDIA InferenceHub key and Astra access without starting Isaac Sim."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward the API credential to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key_file", type=Path, help="Otherwise use NV_API_KEY, then ~/.config/openai/api_key.")
    parser.add_argument("--model", default="openai/openai/gpt-6-astra")
    args = parser.parse_args()
    key = os.environ.get("NV_API_KEY", "").strip() if args.api_key_file is None else ""
    key_file = (args.api_key_file or Path.home() / ".config/openai/api_key").expanduser()
    if args.api_key_file is not None or (not key and key_file.is_file()):
        try:
            key = key_file.read_text().strip()
        except OSError:
            parser.error(f"Cannot read {key_file}; run python3 scripts/tools/configure_astra_key.py")
    if not key:
        parser.error("No key found. Run python3 scripts/tools/configure_astra_key.py or set NV_API_KEY.")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with only OK."}],
        "reasoning_effort": "low",
        "max_completion_tokens": 512,
    }
    request = urllib.request.Request(
        "https://inference-api.nvidia.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=90) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        advice = {
            401: "Check that the key is current and was issued by NVIDIA InferenceHub.",
            403: "Check the key's model permissions and your InferenceHub access.",
            429: "Check your InferenceHub quota/budget and rate limits; retry or contact InferenceHub support.",
        }.get(error.code, "Check the endpoint/model in InferenceHub and service availability.")
        print(f"InferenceHub returned HTTP {error.code}. {advice}")
        return 1
    except (urllib.error.URLError, TimeoutError, OSError):
        print("Could not reach InferenceHub within 90 seconds. Check network access and service availability.")
        return 1
    except (ValueError, UnicodeError):
        print("InferenceHub returned an invalid JSON response.")
        return 1
    choices = document.get("choices") if isinstance(document, dict) else None
    message = (
        choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    )
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        print("InferenceHub responded, but no assistant text was returned. Check model support and token limits.")
        return 1
    print(f"InferenceHub connection OK: {args.model} responded in {time.monotonic() - started:.2f}s.")
    print("The saved key was read locally and was not printed. You can now run the Astra testbed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
