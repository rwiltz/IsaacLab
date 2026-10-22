# Astra environment testbed

Run Isaac Lab environments with Astra planning (System 2) and structured action
inference (System 1). Start with the fixed-base G1 camera example, or use the
generic runner and an explicit environment adapter.

Clone the team branch into a clearly named checkout, then run from its root:

```bash
git clone --branch codex/astra-testbed https://github.com/rwiltz/IsaacLab.git IsaacLab-astra-testbed
cd IsaacLab-astra-testbed
uv sync --extra isaacsim --extra teleop
```

After accepting the NVIDIA Omniverse EULA, set this variable in the terminal used
for simulator commands below. Repeat it in each new terminal:

```bash
export OMNI_KIT_ACCEPT_EULA=YES

# Check G1 simulator, cameras, and action mapping without API requests.
uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
  --device cpu --mock --max_steps 120 --trace logs/astra/g1-mock.jsonl \
  renderer=isaacsim_rtx visualizer=none
```

Create the key before running the configuration helper:

1. Sign in with your NVIDIA account at [InferenceHub](https://inference.nvidia.com),
   then open [Key Management](https://inference.nvidia.com/key-management).
2. Create your own named development key, for example `astra-isaaclab-yourname`,
   choosing the key type and access settings appropriate to your account.
3. Copy the secret into `python3 scripts/tools/configure_astra_key.py`.
4. Run `python3 scripts/tools/check_astra_connection.py --api_key_file ~/.config/openai/api_key` to check the saved key
   and selected Astra model with one short live request, without starting Isaac Sim.

For 401, 403, or 429 errors, check key validity, model access, and quota or rate
limits with InferenceHub support. Its [wiki](https://inference.nvidia.com/wiki)
contains account guidance. The helpers do not print the key.

To run the G1 bin-drop experiment, use this from a desktop terminal. It uses CPU
physics, head and wrist cameras, and eight waypoints of 0.5 simulation seconds
each. RTX camera rendering still requires a supported NVIDIA GPU.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
  --api_key_file "$HOME/.config/openai/api_key" \
  --base_url https://inference-api.nvidia.com/v1 \
  --model openai/openai/gpt-6-astra \
  --api_format chat_completions \
  --task IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs \
  --device cpu \
  --cameras robot_pov_cam,left_wrist_cam,right_wrist_cam \
  --bin_drop --pause_during_inference \
  --action_chunk_size 8 --waypoint_seconds 0.5 \
  --wrist_linear_speed 0.1 --wrist_angular_speed 0.8 --grip_speed 1.6 \
  --system1_timeout 75 --system1_max_age 120 \
  --system2_effort medium --system2_timeout 180 \
  --system2_interval 300 --system2_max_age 360 \
  --reset_on_success --max_episodes 2 \
  --goal "Pick up the steering wheel with the left hand, lift it clear, and transfer support to the right hand before opening and retracting the left. Carry the wheel fully over any reachable part of the bin opening while retaining a secure grasp. Drop it by opening the right hand at a stationary pose; careful placement on the floor or leveling the wheel is unnecessary. If the open thumb remains hooked on the rim, tip the open thumb downward to let gravity release the wheel, keeping the wheel over the bin. Allow actual simulation time for opening, falling and settling. Withdraw clear and verify the wheel rests inside the bin with both hands open and no robot support. Keep the base fixed." \
  --trace "logs/astra/drop-$(date +%Y%m%d-%H%M%S).jsonl" \
  renderer=isaacsim_rtx visualizer=kit
```

The goal is left-hand pickup, handoff to the right hand, then release into the
bin. `--bin_drop` supplies the environment-derived task criteria and checks for
release and settling locally. `--reset_on_success --max_episodes 2` starts a
fresh attempt after verified success and stops after two completed attempts.
Astra generates new actions on each run. The wrist setpoint limits are 0.1 m/s
and 0.8 rad/s, and finger setpoints move at up to 1.6 rad/s. These are command
limits; measured robot motion depends on the controller and contacts.

The shorter equivalent uses the helper's built-in prompt and settings:

```bash
scripts/environments/teleoperation/launch_astra_viewport.sh \
  --reset_on_success --max_episodes 2 \
  --trace "logs/astra/drop-$(date +%Y%m%d-%H%M%S).jsonl"
```

To reproduce the earlier cautious motion settings, override the helper's limits
and waypoint duration:

```bash
scripts/environments/teleoperation/launch_astra_viewport.sh \
  --wrist_linear_speed 0.025 --wrist_angular_speed 0.4 --grip_speed 0.8 \
  --waypoint_seconds 1.0
```

The live helper uses `https://inference-api.nvidia.com/v1`, model
`openai/openai/gpt-6-astra`, and the saved `~/.config/openai/api_key` file. That
historical path does not select the provider. The helper explicitly selects the
provider and key file; append CLI flags to override them. `NV_API_KEY` and
`ASTRA_*` environment defaults apply to direct Python launches. Live requests
include numerical observations and camera images. Override `--goal` and `--trace`
for your experiment.

For a small numerical environment with no camera renderer:

```bash
uv run --no-sync python scripts/environments/teleoperation/run_astra_test.py \
  --task Isaac-Cartpole --adapter astra_teleop.adapters:CartpoleAdapter \
  --goal "Balance the pole near upright and keep the cart near the center." \
  --mock --max_steps 120 --max_episodes 2 \
  --trace logs/astra/cartpole-mock.jsonl \
  physics=isaacsim_physx visualizer=none
```

Remove `--mock` for live inference after checking the integration. The step and
episode flags are upper bounds. A model request can take many control timesteps;
the Cartpole mock is a data-flow test, not a balancing policy.

Both runners can record HDF5 episodes with `--dataset_file`. To save successful
G1 bin-drop episodes, append recording flags to the helper:

```bash
scripts/environments/teleoperation/launch_astra_viewport.sh \
  --reset_on_success --max_episodes 2 \
  --dataset_file "logs/astra/g1-episodes-$(date +%Y%m%d-%H%M%S).hdf5" --dataset_export_mode success
```

`--reset_on_success` starts a fresh attempt after verified success.
`--max_episodes 2` counts completed attempts, including native terminations;
it does not request two saved successful episodes. The timestamped output path
keeps separate launches in separate datasets.

`success` is the default export mode. G1 requires `--bin_drop` in this mode
(already enabled by the helper), so success comes from independent measurements,
not Astra's declaration. Use `--dataset_export_mode all` to retain unsuccessful
native terminations and interrupted or `--max_steps`-limited attempts too; these
are marked unsuccessful. For example, add
`--dataset_file "logs/astra/cartpole-mock-$(date +%Y%m%d-%H%M%S).hdf5" --dataset_export_mode all` to the
Cartpole mock command above, whose adapter has no success detector.

The Isaac Lab recorder stores executed environment actions, processed actions,
initial and step states, and policy observations, retaining the task's policy
observation layout and configured custom recorder terms. Inference pauses add
no step samples. Changing motion speeds or waypoint durations preserves the G1
action recording cadence of 50 Hz during executed simulation steps. Existing
HDF5 recordings retain their recorded actions and replay unchanged. Camera images
enter HDF5 only through task policy observations
or custom recorder terms; use `--image_trace_dir` for inference JPEGs. Choose a
new output path ending in `.hdf5`, or omit the extension to append it
automatically. Existing output files are rejected rather than overwritten.

For another environment, import its registration package with
`--env_module my_project.tasks`, choose `--task My-Registered-Task-v0`, and provide
`--adapter my_project.astra_adapter:MyAdapter`. The adapter owns observations,
command schema and validation, action conversion, and measured task criteria.

- [Setup, generic runner, and adapter guide](../../../docs/source/how-to/astra_testbed.rst)
- [G1 controls, cameras, limits, and bin-drop criteria](../../../docs/source/how-to/astra_teleoperation.rst)
- [Adapter source examples](astra_teleop/adapters.py)

Mock runs verify integration without external inference. On October 21, 2026,
these CPU settings completed a fresh left-hand pickup, right-hand handoff, and
verified bin drop in the `codex/astra-testbed-consolidated` development branch:
44.36 simulation seconds and 611.50 wall seconds. Active execution took
approximately 46.6 wall seconds; inference pauses accounted for the rest. Three
earlier cautious runs in that checkout also completed successfully. These are
scene-specific observations; the faster settings have one completed trial, and
this backport has not repeated that full live experiment.

If a new flag is unrecognized, confirm you are in `IsaacLab-astra-testbed`, not an
older teleoperation worktree. Check the interpreter and imported checkout:

```bash
uv run --no-sync python -c "import sys; from isaaclab.paths import ISAACLAB_ROOT; print(sys.executable); print(ISAACLAB_ROOT)"
```

Both paths should belong to this checkout. Run the sync command here if the
editable environment is stale; `--no-sync` does not update dependencies or paths.

The launchers also normalize non-breaking spaces introduced when copying rich
text. If an older launcher rejects pasted flags, update the checkout or type the
helper directly: `scripts/environments/teleoperation/launch_astra_viewport.sh`.
