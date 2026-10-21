.. _astra-teleoperation:

Astra control of fixed-base G1
=============================

``scripts/environments/teleoperation/teleop_astra_agent.py`` runs the existing
``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs`` environment with GPT-6 Astra
providing both scene planning (System 2) and visual action inference (System 1).
It replaces the input-device side of ``teleop_se3_agent.py``: the environment's
Pink inverse kinematics and ``env.step`` remain the execution path. The robot's
pelvis is fixed. Astra controls the wrist poses and an open/close decision for
each hand; a deterministic controller generates the finger trajectories.

Run the pipeline
----------------

Use a checkout based on ``main`` and its own environment. Install the runtime
and teleoperation dependencies with the repository's uv workflow:

.. code-block:: bash

   uv sync --extra isaacsim --extra teleop

After accepting the NVIDIA Omniverse EULA, set this in the launch terminal:

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES

First check simulator startup, observations, cameras, and action mapping with a
local policy that holds the wrists and opens the hands. This mode makes no API requests:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
       --device cpu --mock --max_steps 250 --trace logs/astra/mock.jsonl \
       renderer=isaacsim_rtx visualizer=none

Set ``NV_API_KEY`` in the launching shell, or provide an owner-readable key
file with ``--api_key_file`` (which takes precedence over the environment).
If neither is supplied, the launcher reads ``~/.config/openai/api_key``.
The standalone helper prompts without displaying
the key and atomically saves it to ``~/.config/openai/api_key`` with mode 0600:

.. code-block:: bash

   python3 scripts/tools/configure_astra_key.py

For the viewport manipulation demonstration, run this script from a desktop terminal.
It locates the checkout and saved key automatically and selects NVIDIA InferenceHub.
Its default goal is to pick up the steering wheel with the left hand, hand it off
to the right hand, and drop it into the bin. It preserves the carried grasp orientation,
opens above the bin, and allocates actual simulation time for falling and settling.
It selects CPU physics and enables verified bin-drop completion, 8-waypoint chunks
with a 0.5-second dwell per waypoint, longer inference timeouts, and physics pauses
during inference waits. RTX cameras and the viewport still use the GPU.
Pass ``--action_chunk_size 30``
for longer chunks. Pass ``--goal`` to
replace this instruction, or append ``--astra_log_level DEBUG`` for waypoint logs:

.. code-block:: bash

   scripts/environments/teleoperation/launch_astra_viewport.sh

For the exact bin-drop prompt and a complete ``uv run`` command with camera,
timing, and reset settings, see :ref:`astra-g1-bin-drop-reproduction`.

For a different task instruction, use the Python launcher directly:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
       --device cpu \
       --api_key_file ~/.config/openai/api_key \
       --goal "Observe the object and move the right hand toward it slowly." \
       --trace logs/astra/reach.jsonl \
       renderer=isaacsim_rtx visualizer=kit

The default provider is NVIDIA InferenceHub, with base URL
``https://inference-api.nvidia.com/v1``, model ``openai/openai/gpt-6-astra``,
and ``--api_format chat_completions``. Both workers use the selected model.
Use a key issued by this provider. The helper's default key-file path is retained
for existing setups; its directory name does not determine the inference provider.

``--model``, ``--base_url``, and ``--api_format`` select another compatible
deployment. The corresponding defaults can also be set with ``ASTRA_MODEL``,
``ASTRA_BASE_URL``, and ``ASTRA_API_FORMAT``. ``--api_key_env`` selects a different
key environment variable. For OpenAI's public API, use an OpenAI-issued key and
``--base_url https://api.openai.com/v1 --model gpt-6-astra --api_format responses``.

The Python launcher's default goal is to hold the wrists, keep the hands open, and observe. Camera
frames and numerical state are sent to the configured API only in live mode.
This script controls one simulated G1, with no XR headset or CloudXR session required.
The launcher disables the training timeout and coarse object-root success reset,
so entering the bin area does not restart the scene before release. The object-drop
termination remains active. Without ``--bin_drop``, Astra's stop decision controls
goal completion. With ``--bin_drop``, a local measured-outcome check controls completion;
the model's stop flag alone never counts as a successful drop.
The environment downloads its usual robot assets and IK description when they
are not cached.

Dropping into the bin and repeating episodes
-------------------------------------------

``--bin_drop`` gives both model roles a structured ``action_spec.task_contract``
derived from the resolved environment success configuration and authored bin geometry.
The original training term tests an object-root box, low componentwise linear velocity,
and right-wrist retraction. It does not establish release or whole-wheel containment,
so it remains disabled. The active replacement clips the wheel mesh at the bin wall
top, including polygon-edge intersections, and checks that all lower geometry lies
inside the bin interior. The mesh bounding-box center must also lie inside the bin's
horizontal bounds. It then checks floor proximity, both commanded and measured
open hands, zero measured per-body robot contact, actual bin support, and low linear/angular
speed for one uninterrupted second of executed simulation time. Above-wall rim
overhang is allowed: a released wheel resting against the bin wall need not have its
entire overhead footprint inside. Geometry protruding outside below wall height still
fails. ``below_wall_xy_margin_m`` reports this clearance, or null when no mesh reaches
below the wall top. The full-footprint ``xy_margin_m`` remains available as guidance.

Each inference snapshot includes ``task_progress`` with these conditions, current
clearances, contact support, and settling time. ``drop_ready`` identifies an above-bin
footprint with at least 2 cm horizontal margin and no more than 35 cm bottom-to-floor
clearance. It is geometric guidance, not a collision-free trajectory or guaranteed
landing. Astra still chooses when to open and where to move; the checker generates
no task motions. Complete finger opening and physical settling dwells before requesting
post-release feedback, because inference pauses freeze the fall too.

The task contract also supplies ``rotation_clearance``: a sphere centered at the
body-frame mesh bounding-box center and enclosing every scaled mesh vertex, plus
the bin-local XY region where that sphere fits. A null region and
``sphere_fits_bin_xy=false`` mean the sphere cannot fit. Measured
``rotation_safe_xy_margin_m`` is nonnegative when the current sphere fits inside
the bin's horizontal bounds. This conservatively contains the mesh footprint at
any orientation only while the sphere center stays fixed. Rotation about the wrist
or a contact can translate that center; falling and bounce can move it too. This
guidance changes neither ``drop_ready`` nor the success criteria.

Contact support is unavailable immediately after construction or reset until physics
executes; raw PhysX buffers can otherwise retain forces from the preceding episode.
``robot_contacts`` names the contacting robot bodies, including palms, and provides
their force vectors separately from ``bin_contact_force_world_n``. The 0.001 N
reporting threshold omits negligible entries from this list; the zero-contact success check still
uses every body force, including omitted entries.
Preserving orientation applies to carrying and initial opening. If the rim remains
hooked on the fully open thumb or palm, Astra can choose a release orientation or
combined clearance motion over the bin and inspect the resulting contact change.
The hand-frame guide describes the open thumb direction. A thumb pointing upward can
retain the rim; the release prompt asks Astra to tip it downward so gravity can help,
using bounded waypoints for the required net rotation and checking contacts along the way.
Repeated translation that merely carries the caught wheel is not a successful release.
Persistent wrist tracking error calls for a revised approach or reachable pose; a
fully contained drop near the bin edge is acceptable and need not target its center.

The helper exits after a verified drop. To reset after success and compare two complete
attempts in the same process:

.. code-block:: bash

   OMNI_KIT_ACCEPT_EULA=YES scripts/environments/teleoperation/launch_astra_viewport.sh \
       --reset_on_success --max_episodes 2 \
       --trace logs/astra/drop-repeat.jsonl

``--max_episodes`` counts completed attempts, including environment terminations;
an object-drop termination is not reported as success. Each ``episode_completed``
trace event reports the verified outcome, wall/simulation duration, time from ready
state to first action, submitted request counts, accepted S1/S2 latencies, and feedback
count. These durations exclude environment construction and reset/setup. Any observed
late reply from the preceding episode is recorded separately and cannot execute.

There is no cross-episode policy memory or automatic weight update: plans, chunks,
and temporal comparisons reset. A repeated attempt may benefit from warmed runtime
resources or provider input caching, but faster inference or fewer decisions are not
guaranteed. Stable task metadata precedes changing observations in requests to improve
prefix reuse where the provider supports it. Compare successful full episodes separately
from restored near-bin release probes, and do not interpret a second fast release probe
as faster pickup and handoff.

Inference and simulation timing
-------------------------------

The simulation thread targets the environment's 50 Hz control rate when calling
``env.step`` (200 Hz physics with decimation 4), paced against a monotonic wall clock.
The launcher renders once per executed control step, or 50 Hz in simulation time,
while preserving the 200 Hz physics rate. Render-only updates during inference
pauses do not advance physics or execute actions.
Rendering and simulation cost can reduce the actual wall-clock rate.
Two independent inference workers operate on detached observation snapshots:

* System 2 uses high reasoning effort to produce a subgoal, success criteria,
  and a stop decision. Its default minimum request interval is 10 seconds.
* System 1 uses low reasoning effort, the latest plan, numerical observations,
  and camera frames to produce a bounded command or chunk. Its default minimum request
  interval is 1 second. No action inference starts before a valid plan arrives.

``--system1_effort`` and ``--system2_effort`` override these reasoning levels.
The viewport helper uses medium planning effort, matching the release experiments;
the general launcher's default remains high. Measured inference times vary by state
and request, so this setting does not guarantee a particular speed or success rate.
The viewport helper allows up to 180 wall seconds for planning; three-view
recovery assessments have exceeded 120 seconds. This raises the timeout, not the
expected inference speed. The general launcher's default remains 60 seconds.
The helper spaces periodic planning requests by 300 seconds, with a 360-second
plan age limit. Explicit action feedback checkpoints still request a fresh assessment
immediately. This avoids starting another periodic assessment as soon as a slow
planning reply arrives, only to discard it at the next contact checkpoint.

These are request intervals, not guaranteed model response rates. Astra does
not run the joint servo loop. The existing controllers execute accepted commands
while inference is pending. At most one request per worker is in flight; there
is no accumulating queue of old camera frames.

Add ``--pause_during_inference`` to freeze physics whenever a request is pending
and no usable action is ready. The viewport remains responsive and inference
continues; robot motion, grip trajectories, and waypoint dwell clocks resume only
when simulation stepping resumes. Active chunks continue executing while their
successors are prefetched. Request timeouts and snapshot ages still use wall time.
This option is disabled by default. ``--max_steps`` counts executed control steps,
so it does not impose a wall-time limit on inference waits. Console messages report
pause and resume transitions. Astra receives this timing mode in its action
specification: inference waiting does not count as settling or a stable hold.
It must allocate explicit unchanged waypoints for those simulation-time checks.

``--system1_interval``, ``--system2_interval``, ``--system1_timeout``, and
``--system2_timeout`` control request scheduling and timeouts in wall-clock
seconds. ``--system1_max_age`` (default 10 seconds) and ``--system2_max_age``
(default 30 seconds) reject decisions based on old snapshots. An action lasts
at most ``--max_hold_seconds`` (default 1 second) in single-command mode. Tune these values using the
trace and measured API latency; conservative defaults can produce intermittent
motion when inference is slow.

A new plan preempts commands from the previous plan. Episode resets invalidate
pending decisions. On entering a chunk-mode inference wait with a valid active
plan, the simulation freezes each wrist at its last commanded setpoint. Any
unfinished movement toward the old endpoint ends with its allocated dwell.
The hold remains fixed as measured wrists settle or drift under load, while
finger trajectories continue toward their accepted targets. A changed plan
reanchors the wrist hold once at its measured pose, retaining the finger targets
until a new action.
An explicit stop, an expired plan, or single-command expiry cancels the finger
trajectory and captures a measured wrist and finger hold once. Subsequent idle
ticks retain those fixed setpoints even if the measured robot drifts. A new
accepted action resumes control; an episode reset captures a new initial hold.
This is a commanded hold, not a guarantee of successful
manipulation.

Action chunking
---------------

Set ``--action_chunk_size 30 --waypoint_seconds 1.0`` to request 30 wrist/grip
waypoints at once. Each target is applied for 1 simulation second: a chunk
covers 30 simulation seconds, or 1500 control steps at 50 Hz. These are timed
targets, not a promise that inverse kinematics reaches every target on schedule.
Thirty individual 50 Hz control ticks would cover only 0.6 seconds.

Once a chunk starts, System 1 can prepare its successor while the simulator
executes the current chunk. The request includes fresh observations and camera
frames, the remaining commands, and the predicted end target. Predicted targets
are distinguished from measured state. One active chunk and one prefetched chunk
are retained; a completed reply never replaces a chunk partway through execution.
Ready chunks switch without an idle control tick between them. If inference falls
behind, the robot holds until a usable chunk arrives.

For phases that require new visual feedback, System 1 sets
``feedback_after_waypoints`` to a positive count. After that many complete dwells,
the pipeline discards the chunk's remaining tail and requests System 2 from fresh
observations. It suppresses successor prefetch for this chunk and rejects planner
replies captured before the checkpoint. Action inference resumes after the fresh
assessment. The prompt places these checkpoints at phase endpoints or contact
boundaries, including completed alignment, closure, and test lifts. A known clear
rotation, retraction, or approach may continue across several chunks with a zero
count and continuous prefetch until its endpoint is reached. The end of an array
alone does not require a checkpoint. Console logs show when a checkpoint is
reached and when its assessment is ready.

For example, launch with more time allowed for the longer response:

.. code-block:: bash

   uv run --no-sync python scripts/environments/teleoperation/teleop_astra_agent.py \
       --action_chunk_size 30 --waypoint_seconds 1.0 \
       --system1_timeout 75 --system1_max_age 120 \
       --system2_interval 120 --system2_max_age 240 \
       --goal "Keep wrists stationary. Slowly close both hands, then open them." \
       --astra_log_level DEBUG --trace logs/astra/chunks.jsonl \
       renderer=isaacsim_rtx visualizer=kit

Chunk execution advances only after successful ``env.step`` calls, using the
environment's simulation timestep. Request timeouts and observation age limits
still use wall time. In chunk mode, ``--system1_max_age`` is checked when a reply
arrives and again when its queued chunk starts; it does not expire every waypoint
of an active chunk. The fixed chunk length bounds execution duration.
``--max_hold_seconds`` applies only to single-command mode (the default,
``--action_chunk_size 1``).

A changed or expired plan, a stop decision, or an episode reset clears both
chunks and invalidates dependent requests. A failed or stale prefetch leaves an
already accepted active chunk running; the buffer holds if its successor is not
ready. Chunking trades more motion between observations for fewer inference gaps.
It cannot guarantee continuous execution if generation takes longer than a chunk.
In an earlier benchmark, one NVIDIA InferenceHub probe with the actual scene camera took 43 seconds for
30 waypoints. Simulator measurements ranged from 16 to 30 control steps per wall
second, so a 30-waypoint chunk with one-second dwell can last roughly 50 to 94 wall
seconds. These are historical measurements from before the current CPU and
rendering settings. The example allows a queued chunk to age for up to 120 wall seconds and
spaces System 2 requests 120 seconds apart. Frequent changed plans can otherwise
discard slow System 1 requests before they finish. These settings trade more
latency tolerance for older observations and less frequent high-level reassessment.
Use console timings to tune them; one API measurement is not a latency guarantee.
Use short, bounded moves around contact: a grasp or handoff needs fresh feedback
before proceeding to the next stage. In the ``codex/astra-testbed-consolidated``
development branch, these CPU settings completed one fresh pickup, handoff, and
verified bin drop in 44.36 simulation seconds and 611.50 wall seconds on
October 21, 2026. See :doc:`astra_testbed` for the validation scope and comparison
with earlier runs.

Observations and action contract
--------------------------------

The adapter passes named numerical policy observations and joint names, along
with current wrist and hand poses. A separate object contact sensor reports
forces on the wheel from individually named finger bodies; table contacts do
not contribute to those columns. The existing hand sensors still report all
contacts, which helps distinguish touching the table from touching the wheel.
Neither contact signal alone establishes a secure grasp.

The adapter also supplies cached mesh bounds from the composed USD asset in
the object's rigid-body frame, including asset scale. Authored rim geometry
includes its radial and height ranges when an outer vertex band is identifiable.
This is known asset geometry, separate from camera observations and collision
measurements; it supplies the offset between the object root and the rim.
Compact motion observations retain the measured object pose at episode start and
compare object and wrist motion with the preceding inference snapshot. Relative
object motion in each wrist frame helps assess capture, but a grasped rim can
pivot under gravity without leaving the fingers. Root-pose drift alone does not
establish grasp loss; contact-region retention, current support forces, and fresh
camera views help distinguish pivoting from slipping. Let the object settle
before transfer. These are discrete measurements, not proof that a grasp remained
stable between snapshots. ``contact_centroid_translation_wrist_m`` reports each
finger's mean contact-position change in its associated wrist frame, when both
adjacent inference snapshots have finite contact positions. A rigid hand
translation alone produces no change in that local contact position. The points
are unweighted contact-region centroids, not tracked material points; lost or
new contacts have no displacement entry.

The adapter also reads the authored bin's floor and wall colliders from the
packing-table asset. It reports the bin pose and interior bounds in the bin frame,
with source labels and environment bounds, so placement can account for the
whole wheel rather than only its root position.
Dense link-state and camera tensors are
excluded from the JSON observations. ``--cameras`` takes a comma-separated list
of RGB sensor names; the default is ``robot_pov_cam,left_wrist_cam,right_wrist_cam``.
The launcher adds virtual RGB cameras to both wrist bodies. They move with the
hands and look toward the fingers, using a wide field of view and a 1 cm near
clipping plane for close contact. Astra receives all three named views in each
request: the head camera for the scene and the wrist cameras for finger/rim
alignment. These mounts are virtual viewpoints, not physical camera calibrations.
Select ``--cameras robot_pov_cam`` to run with only the head camera.

In the Kit viewport's camera selector, the wrist views are named ``AstraWristCam``
under ``Robot/left_wrist_yaw_link`` and ``Robot/right_wrist_yaw_link``. The external
Perspective viewport used for recording is independent of the sensor views sent
to Astra. Images are copied on the
simulation thread, resized to a maximum edge of ``--image_size`` (default 640 pixels),
and sent as JPEG image inputs. The camera is refreshed after environment resets
before new observations are submitted for inference.

System 1 returns a strict JSON object containing ``command``, ``hold_seconds``,
and ``stop`` in single-command mode. Chunk mode returns ``commands`` (exactly
``--action_chunk_size`` commands), ``feedback_after_waypoints`` (an integer from
zero through the chunk size), and ``stop``. Every waypoint is validated before
any part of a chunk is accepted. Each command has these fields:

.. list-table:: G1 command fields
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``left_wrist``, ``right_wrist``
     - ``position`` [m] relative to the environment origin in world axes;
       ``quaternion`` in XYZW order, also in world axes.
   * - ``left_grip_closed``, ``right_grip_closed``
     - Boolean grip goals: ``false`` opens the hand; ``true`` closes it.
       The adapter generates seven finger joint targets per hand.

The adapter constructs the environment's 28-value action in its resolved order:
left wrist (7), right wrist (7), and hand joints (14).
Hand joint ordering comes from the articulation and is explicitly mapped from
the trajectory's semantic order. Wrist coordinates are **not** pelvis-relative;
the existing Pink action handles that conversion.

Each accepted command may move wrists at most 0.05 m and rotate them at most
0.2 rad from measured state. A trajectory advances wrist setpoints each control
tick at no more than 0.1 m/s and 0.8 rad/s, using shortest-arc quaternion
interpolation. ``--wrist_linear_speed`` and ``--wrist_angular_speed`` override
these limits. They limit commanded velocity, not actual acceleration or tracking
error. Astra receives these limits and must allocate enough waypoint dwell for
the move: a 5 cm destination takes at least 0.5 seconds at the default speed.
Wrist interpolation runs during the command's allotted execution time. At a
chunk feedback checkpoint or buffer underrun, it freezes at the last commanded
setpoint. The finger controller advances toward the selected open or closed pose
on every simulation control step, including ordinary chunk inference waits,
capped at 1.6 rad/s per joint and the articulation's joint limits.
``--grip_speed`` overrides the finger speed limit. An expired
plan, a single-command expiry, or an explicit stop instead holds measured wrists
and fingers at the pose captured on entering idle and cancels the finger
trajectory. Idle ticks reuse that captured pose rather than following measured
drift.
For the earlier cautious motion settings, use ``--wrist_linear_speed 0.025
--wrist_angular_speed 0.4 --grip_speed 0.8 --waypoint_seconds 1.0``.
Unsupported action layouts fail at initialization.
Other robots need their own adapter with a defined observation/action contract.

The fixed grip trajectory uses zero joint angles for open and a 90% closed
preset derived from the TriHand retargeter's flexion targets, with mirrored
left/right joint signs. Both poses are clamped to the simulated joint limits.
In a headless calibration, a deeper rim grasp with this preset and a slow lift
held the whole wheel about 3.4 cm above the table for two simulation seconds.
That verifies a physical grasp configuration, not Astra's complete pickup,
handoff, and placement policy. The poses and speed are defined in
``scripts/environments/teleoperation/astra_teleop/g1.py``.

The action specification also exposes approximate nominal pinch centers for
this closed preset, in each wrist's local frame: left
``[0.12125379, -0.08139178, -0.00274833]`` m and right
``[0.12125379, +0.08139178, -0.00274833]`` m. The left offset comes from the
simulator calibration. The mirrored right offset was also tested: its grasp
held the whole wheel about 4.7 cm above the table with three right-finger
contacts and less than one micrometer of relative translation over a two-second
fixed hold. These are robot-frame tool offsets. A candidate grasp still requires
selecting a surface
from the current scene and verifying actual capture through cameras and contacts.
The center's environment position is the wrist position plus its rotated offset.

Diagnostics
-----------

Console logging is enabled by default. It reports request submission, response
latency, plans, chunk starts/completions, discarded results, and buffer underruns.
Every five wall seconds a progress line shows the step rate, simulation time,
current waypoint, queued chunk, and pending workers. ``--status_interval`` changes
this interval; zero disables periodic progress.

Use ``--astra_log_level DEBUG`` to also see each waypoint's requested wrist/grip
targets and the wrist targets applied after the adapter's bounds. This flag does
not turn on verbose simulator or HTTP logging. ``WARNING`` and ``ERROR`` reduce
console output. Console messages exclude credentials, camera bytes, and raw HTTP
error bodies.

``--trace`` writes JSONL numerical snapshots, inference events, status, and the
actual action sent to ``env.step``. Render-only inference waits are marked
``inference_paused: true`` with ``action: null``; they do not represent control
steps. Unchanged paused iterations are omitted. The trace omits API keys and image bytes. Use
``--image_trace_dir logs/astra/images`` to save the exact JPEG views captured for
inference, named by episode, step, capture sequence, and camera. The trace records
``capture_id`` so multiple captures at a frozen physics step remain distinct. Compare these images with the
matching snapshot in the trace when diagnosing alignment or occlusion. Use a
new output directory for each run to preserve earlier images. Use
``--max_steps`` to bound a smoke test; Ctrl+C stops an interactive run.
Mock mode verifies the plumbing but does not establish live API latency,
manipulation quality, or task success.

If ``nvidia-smi`` works in a host terminal but fails in a restricted execution
environment, check GPU device access for that process before changing drivers.
Both CUDA and RTX rendering need access to the host GPU.

The request format follows the official OpenAI documentation for
`GPT-6 Astra <https://developers.openai.com/api/docs/models/gpt-6-astra>`_,
`image inputs <https://developers.openai.com/api/docs/guides/images-vision>`_, and
`structured outputs <https://developers.openai.com/api/docs/guides/structured-outputs>`_.
