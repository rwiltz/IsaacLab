#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Launch the fixed-base G1 demo with the saved NVIDIA InferenceHub key.
set -euo pipefail

astra_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
astra_repo_root="$(cd -- "${astra_script_dir}/../../.." && pwd)"
cd -- "${astra_repo_root}"

exec uv run --no-sync python "${astra_script_dir}/teleop_astra_agent.py" \
    --api_key_file "${HOME}/.config/openai/api_key" \
    --base_url https://inference-api.nvidia.com/v1 \
    --model openai/openai/gpt-6-astra \
    --api_format chat_completions \
    --device cpu \
    --action_chunk_size 8 --waypoint_seconds 0.5 \
    --wrist_linear_speed 0.1 --wrist_angular_speed 0.8 --grip_speed 1.6 \
    --pause_during_inference \
    --bin_drop \
    --system1_timeout 75 --system1_max_age 120 \
    --system2_effort medium --system2_timeout 180 \
    --system2_interval 300 --system2_max_age 360 \
    --goal "Pick up the steering wheel with the left hand, lift it clear, and transfer support to the right hand before opening and retracting the left. Carry the wheel fully over any reachable part of the bin opening while retaining a secure grasp. Drop it by opening the right hand at a stationary pose; careful placement on the floor or leveling the wheel is unnecessary. If the open thumb remains hooked on the rim, tip the open thumb downward to let gravity release the wheel, keeping the wheel over the bin. Allow actual simulation time for opening, falling and settling. Withdraw clear and verify the wheel rests inside the bin with both hands open and no robot support. Keep the base fixed." \
    --trace "${astra_repo_root}/logs/astra/demo.jsonl" \
    renderer=isaacsim_rtx visualizer=kit "$@"
