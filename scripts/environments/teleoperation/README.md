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
  --mock --max_steps 120 --trace logs/astra/g1-mock.jsonl \
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

Then launch the live G1 demo from a desktop terminal:

```bash
scripts/environments/teleoperation/launch_astra_viewport.sh
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

For another environment, import its registration package with
`--env_module my_project.tasks`, choose `--task My-Registered-Task-v0`, and provide
`--adapter my_project.astra_adapter:MyAdapter`. The adapter owns observations,
command schema and validation, action conversion, and measured task criteria.

- [Setup, generic runner, and adapter guide](../../../docs/source/how-to/astra_testbed.rst)
- [G1 controls, cameras, limits, and bin-drop criteria](../../../docs/source/how-to/astra_teleoperation.rst)
- [Adapter source examples](astra_teleop/adapters.py)

Mock runs verify integration without external inference. Prior G1 release evidence
came from a restored near-bin state and a separate physical hold verification;
fresh full-task completion and repeated Astra episodes remain unverified.

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
