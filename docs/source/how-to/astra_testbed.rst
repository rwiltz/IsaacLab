.. _astra-testbed:

Testing environments with Astra
===============================

The Astra testbed connects an Isaac Lab environment to two model workers: System 2
turns a goal and observations into a subgoal, and System 1 produces structured
actions for that subgoal. The simulator executes validated actions through
``env.step``. Numerical observations are sufficient; an adapter can also supply
camera frames and task measurements.

Use ``run_astra_test.py`` for an environment with an explicit adapter. The
fixed-base G1 demonstration has its own launcher, ``teleop_astra_agent.py``, which
adds wrist cameras, inverse kinematics, finger trajectories, and manipulation
measurements. Both use the same inference and scheduling core. Start with the
G1 example to inspect a camera-based robot workflow, or Cartpole to exercise the
adapter interface with a small numerical action space.

Prepare a checkout
------------------

Use the team's ``codex/astra-testbed`` branch, based on ``main``. From a new clone:

.. code-block:: bash

   git clone --branch codex/astra-testbed https://github.com/rwiltz/IsaacLab.git IsaacLab-astra-testbed
   cd IsaacLab-astra-testbed
   uv sync --extra isaacsim --extra teleop

For an existing clone, fetch the team fork explicitly and create a separate
worktree. This works even when ``origin`` points to upstream Isaac Lab. Choose an
unused experiment branch name and destination directory:

.. code-block:: bash

   git fetch https://github.com/rwiltz/IsaacLab.git codex/astra-testbed
   git worktree add -b codex/astra-experiment-yourname ../IsaacLab-astra-testbed FETCH_HEAD
   cd ../IsaacLab-astra-testbed
   uv sync --extra isaacsim --extra teleop

After accepting the NVIDIA Omniverse EULA, set this variable in the terminal
used for all simulator commands below, including launcher help:

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES

Repeat the export when opening a new terminal. The standalone key and connection
helpers do not launch the simulator and do not require this variable.

The examples below run from the new ``IsaacLab-astra-testbed`` checkout root.
They currently support one manager-based RL environment at a time. The simulator needs a supported
NVIDIA GPU and working host driver access. First launch may download environment
assets and initialize rendering caches. The G1 example uses Isaac Sim's RTX
renderer; the Cartpole example needs no camera renderer or desktop viewport.
The repository installation guide covers platform and simulator prerequisites.

Check the G1 integration locally
--------------------------------

Run a bounded test without an inference key:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
       --mock --max_steps 120 --trace logs/astra/g1-mock.jsonl \
       renderer=isaacsim_rtx visualizer=none

Mock mode uses the same simulator and action validation, with a local client that
repeats the adapter's neutral command. For G1 this holds the wrists and opens the
hands. It verifies startup and data flow, not live model quality or manipulation
success. No model API requests are made.

Configure live inference
------------------------

Create and check your own NVIDIA InferenceHub development key:

#. Sign in to `InferenceHub <https://inference.nvidia.com>`_ with your NVIDIA
   account, then open `Key Management <https://inference.nvidia.com/key-management>`_.
#. Create a key with a recognizable development name, such as
   ``astra-isaaclab-yourname``. Select the key type and access settings appropriate
   to your account in the portal; available choices depend on your access.
#. Copy the secret into the hidden prompt from the local helper:

   .. code-block:: bash

      python3 scripts/tools/configure_astra_key.py

#. Check the saved key and the selected Astra deployment before starting the
   simulator:

   .. code-block:: bash

      python3 scripts/tools/check_astra_connection.py --api_key_file ~/.config/openai/api_key

The helpers run without Isaac Sim. The connection check sends one short live
request to the NVIDIA endpoint, consuming normal inference quota, then prints
the selected model and latency. It prints neither the key nor the response body. If it reports 401, 403, or 429, check key validity,
model permissions, and quota or rate limits with InferenceHub support. Consult
its `wiki <https://inference.nvidia.com/wiki>`_ for current account guidance.

The configuration helper writes ``~/.config/openai/api_key`` with owner-only
permissions. The directory name is historical; the default deployment is NVIDIA InferenceHub at
``https://inference-api.nvidia.com/v1``, with model
``openai/openai/gpt-6-astra`` and ``--api_format chat_completions``. Use a key
issued for that provider. The viewport helper explicitly uses this saved file.
For direct Python launches, ``NV_API_KEY`` is an alternative; an explicit
``--api_key_file`` takes precedence. Do not put keys in scripts, traces, or
version control.

.. _astra-g1-bin-drop-reproduction:

Reproduce the G1 bin-drop experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From a desktop terminal, use the same prompt, head and wrist cameras,
action-chunk timing, and inference settings as the bin-drop branch:

.. code-block:: bash

   OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
     --api_key_file "$HOME/.config/openai/api_key" \
     --base_url https://inference-api.nvidia.com/v1 \
     --model openai/openai/gpt-6-astra \
     --api_format chat_completions \
     --task IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs \
     --cameras robot_pov_cam,left_wrist_cam,right_wrist_cam \
     --bin_drop --pause_during_inference \
     --action_chunk_size 8 --waypoint_seconds 1 \
     --system1_timeout 75 --system1_max_age 120 \
     --system2_effort medium --system2_timeout 180 \
     --system2_interval 300 --system2_max_age 360 \
     --reset_on_success --max_episodes 2 \
     --goal "Pick up the steering wheel with the left hand, lift it clear, and transfer support to the right hand before opening and retracting the left. Carry the wheel fully over any reachable part of the bin opening while retaining a secure grasp. Drop it by opening the right hand at a stationary pose; careful placement on the floor or leveling the wheel is unnecessary. If the open thumb remains hooked on the rim, tip the open thumb downward to let gravity release the wheel, keeping the wheel over the bin. Allow actual simulation time for opening, falling and settling. Withdraw clear and verify the wheel rests inside the bin with both hands open and no robot support. Keep the base fixed." \
     --trace "logs/astra/drop-$(date +%Y%m%d-%H%M%S).jsonl" \
     renderer=isaacsim_rtx visualizer=kit

The goal covers left-hand pickup, transfer to the right hand, and dropping into
the bin. ``--bin_drop`` supplies environment-derived task criteria and checks for
release and settling locally. ``--reset_on_success --max_episodes 2`` starts a
fresh attempt after verified success and stops after two completed attempts.
This reproduces the setup; Astra generates new actions on each run.

The viewport helper contains the same prompt and settings, so the shorter
equivalent is:

.. code-block:: bash

   scripts/environments/teleoperation/launch_astra_viewport.sh \
       --reset_on_success --max_episodes 2 \
       --trace "logs/astra/drop-$(date +%Y%m%d-%H%M%S).jsonl"

Override the goal and output path for a smaller experiment:

.. code-block:: bash

   scripts/environments/teleoperation/launch_astra_viewport.sh \
       --goal "Keep both wrists stationary, open both hands, and observe the scene." \
       --trace logs/astra/g1-hold.jsonl

Live mode sends the adapter's numerical snapshot and selected camera images to
the configured provider. The default model is shared by both workers. Direct
Python launches accept ``--base_url``, ``--model``, and ``--api_format``, or their
``ASTRA_BASE_URL``, ``ASTRA_MODEL``, and ``ASTRA_API_FORMAT`` environment defaults.
The viewport helper explicitly selects NVIDIA and its saved key file; append
CLI flags to override those choices, rather than setting environment defaults.
See :doc:`astra_teleoperation` for G1 controls, camera selection, trajectory
limits, and bin-drop measurement details.

Run an environment through an adapter
-------------------------------------

The generic runner imports a registered task and an adapter selected by
``module:Class``. Try the included Cartpole adapter first:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/run_astra_test.py \
       --task Isaac-Cartpole --adapter astra_teleop.adapters:CartpoleAdapter \
       --goal "Balance the pole near upright and keep the cart near the center." \
       --mock --max_steps 120 --max_episodes 2 \
       --trace logs/astra/cartpole-mock.jsonl \
       physics=isaacsim_physx visualizer=none

``--max_steps`` bounds executed control steps; ``--max_episodes`` bounds completed
episodes. Reaching either limit ends the run, so this command does not promise
two episodes within 120 steps. Cartpole uses numerical state and a scalar action;
no images or RTX renderer are needed. Its mock command is a plumbing check, not
a balancing policy.

After the mock test, try a bounded live run with numerical observations only:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/run_astra_test.py \
       --task Isaac-Cartpole --adapter astra_teleop.adapters:CartpoleAdapter \
       --goal "Balance the pole near upright and keep the cart near the center." \
       --pause_during_inference --max_steps 120 --max_episodes 2 \
       --trace logs/astra/cartpole-live.jsonl \
       physics=isaacsim_physx visualizer=none

The generic runner defaults to eight commands per chunk and 0.1 simulation
seconds per command. Cartpole commands are ``{"cart_force": value}``, with value
in [-1, 1]; the environment's existing action scale maps that to [-100, 100] N.
The neutral command is zero force. Native five-second timeout and cart-bound
terminations remain active; the adapter adds no success detector. The live
example checks model integration, not expected reliable balancing: a model call
can take much longer than a control timestep. Choose an action abstraction and
timing suitable for each task. Direct model calls do not replace a
high-frequency stabilizing controller.

Bring your own environment
--------------------------

Register the task with Gymnasium in an importable Python module. Load that
registration module with ``--env_module`` and select your adapter with
``--adapter``:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/run_astra_test.py \
       --env_module my_project.tasks \
       --task My-Registered-Task-v0 --adapter my_project.astra_adapter:MyAdapter \
       --goal "Perform the task described by this environment's measured criteria." \
       --mock --max_steps 120 --trace logs/astra/custom-mock.jsonl \
       visualizer=none

Select physics and rendering presets supported by your task. The placeholder
command uses the task's default backend; add a ``physics=...`` override only when
that preset is available. Cartpole's renderer is already absent by default, and
it does not accept a ``renderer=none`` preset override.

Install your package into the checkout's environment or otherwise make it
importable. Repeat ``--env_module`` to import multiple registration modules.
Registration supplies the task configuration; the adapter supplies
the model-facing observations, action semantics, and completion criteria. An
existing Gymnasium task name alone does not define how an Astra command maps to
that task's action tensor. Use the included ``CartpoleAdapter`` as the minimal
source example and retain environment-specific bounds in your own adapter.

The adapter implements this interface; methods run on the simulation thread
unless stated otherwise:

.. list-table:: Adapter interface
   :header-rows: 1
   :widths: 48 52

   * - Member
     - Responsibility
   * - ``configure_env(env_cfg, args)`` (class method)
     - Configure the environment before construction, including any sensors.
   * - ``__init__(env)`` and ``contract``
     - Bind the constructed environment and expose an ``ActionContract``.
   * - ``describe()``
     - Return JSON-compatible action semantics and task criteria for the model.
   * - ``reset()``
     - Clear adapter state for a new episode.
   * - ``snapshot(observations, episode_id, step, sim_time)``
     - Return detached JSON-compatible measurements and ``neutral_command``.
   * - ``capture_images(cameras, image_size)``
     - Return image inputs for selected sensors, or an empty list without cameras.
   * - ``action(command)`` and ``idle_action()``
     - Return the environment's action tensor for an accepted command or a wait.
   * - ``task_progress(dt)``
     - Return measured task progress or ``None``; count physical dwell using
       executed simulation time. Pauses reuse the preceding measured progress.

An action contract defines the command object independently of the robot:

.. code-block:: python

   from astra_teleop.core import ActionContract

   contract = ActionContract(
       command_schema={
           "type": "object",
           "properties": {"effort": {"type": "number", "minimum": -1.0, "maximum": 1.0}},
           "required": ["effort"],
           "additionalProperties": False,
       },
       instructions="Use effort to control the task; describe its units and sign here.",
   )

The runner injects the adapter's contract into the shared pipeline. Its schema constrains
structured model output and local validation; its instructions replace the G1
wrist/grip assumptions. An optional ``command_validator`` adds semantic checks
that JSON Schema cannot express. This callback must be pure and thread-safe,
without simulator or tensor access. Use an inline command schema: schema
references and IDs are unsupported because the command is embedded in the
response envelope. The provider's structured-output schema restrictions also
apply. Simulator-specific validation and tensor
construction remain on the simulation thread. Mock inference repeats
``snapshot.neutral_command``, which must satisfy the same contract.

Define task success from measured environment state in the adapter, and expose
those criteria to the model. A valid action or a model ``stop`` decision does not
by itself establish successful task completion. ``--reset_on_success`` requests
an explicit reset after adapter-verified success; native environment terminations
still trigger their normal automatic resets. ``--max_episodes`` caps completed
episodes from these paths. Reset adapter state with each episode, including held
commands and any accumulated settling time. Retain a bounded neutral action for
inference waits and invalid decisions.

The generic runner defaults to no cameras. If your adapter configures RGB
sensors, select their names with ``--cameras camera_a,camera_b`` and a compatible
renderer preset. Camera setup and interpretation remain the adapter's
responsibility; importing a task does not automatically install the G1 wrist
cameras or G1 manipulation criteria.

Inspect and compare runs
------------------------

Use separate trace paths for experiments. Without ``--trace``, the generic
runner creates a timestamped file under ``logs/astra``. JSONL traces record
numerical snapshots, pipeline events, and applied actions; credentials and image
bytes are excluded. Both runners support ``--image_trace_dir`` for matched camera
JPEGs when cameras are selected. ``--astra_log_level DEBUG`` adds diagnostic
logging; ``--status_interval`` controls periodic progress messages.
Model response latency is a wall-time measurement. Executed simulation time
advances with successful ``env.step`` calls; render-only inference pauses do not
supply physical motion or settling time.

The G1 guide documents action chunking, checkpoint feedback, and optional physics
pauses while waiting for inference. Those scheduling choices affect observation
age and contact feedback. Tune them from measured latency and task dynamics,
then evaluate success separately from action validity or simulator startup.

The prior G1 work demonstrated calibrated physical grasps and an Astra-generated
release from a restored near-bin state. A corrected geometry check and a separate
no-API restored hold verified that released wheel resting in the bin. That is
limited evidence for that scene and state. Fresh pickup, handoff, drop, and
repeated complete episodes have not been validated as a reliable general policy.
The testbed makes those experiments repeatable; it does not establish success on
an arbitrary registered environment.

Check the active checkout
-------------------------

An ``unrecognized arguments`` error for a new flag can mean the command is
running an older checkout. Confirm the working directory and imported source
before changing the command:

.. code-block:: bash

   cd /path/to/IsaacLab-astra-testbed
   git branch --show-current
   uv run --no-sync python -c "import sys; from isaaclab.paths import ISAACLAB_ROOT; print(sys.executable); print(ISAACLAB_ROOT)"
   uv run --no-sync python scripts/environments/teleoperation/run_astra_test.py \
       --task Isaac-Cartpole --adapter astra_teleop.adapters:CartpoleAdapter \
       --goal "Check available options" --help

The interpreter and ``ISAACLAB_ROOT`` should belong to this worktree. An older
``IsaacLab-astra-teleop`` checkout may lack the bin-drop and episode flags. Each
worktree needs its own environment; run ``uv sync --extra isaacsim --extra teleop``
in the intended checkout after dependency changes. ``--no-sync`` deliberately
reuses the installed environment and cannot repair stale editable-package paths.

Copied commands can also contain non-breaking spaces from a document or chat.
The Astra launchers remove this padding at argument edges, while preserving
quoted goal contents and an explicit empty camera selection. If your checkout
predates that handling, update it or type the helper command directly:

.. code-block:: bash

   scripts/environments/teleoperation/launch_astra_viewport.sh --bin_drop --reset_on_success --max_episodes 2
