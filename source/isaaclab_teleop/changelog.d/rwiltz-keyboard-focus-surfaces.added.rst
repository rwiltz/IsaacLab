* Added :func:`~isaaclab_teleop.se2_keyboard_teleop_cfg`, :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`,
  :func:`~isaaclab_teleop.se2_gamepad_teleop_cfg`, :func:`~isaaclab_teleop.se3_gamepad_teleop_cfg`,
  :func:`~isaaclab_teleop.se2_spacemouse_teleop_cfg` and :func:`~isaaclab_teleop.se3_spacemouse_teleop_cfg`,
  which build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard, gamepad or SpaceMouse
  SE(2)/SE(3) teleoperation. The keyboard is read from whichever visualizer window has focus (no
  ``input`` group, no keys from other apps) and the gamepad from ``/dev/input/jsN``, in process; the
  SpaceMouse is read from ``/dev/hidrawN`` (which needs a udev rule) by the Isaac Capture
  ``spacemouse`` plugin, which the SpaceMouse configs start with the session. Without ``--xr`` they
  run headset-free on the standalone CloudXR profile.
* Added :mod:`isaaclab_teleop.device_selection`, which translates an :mod:`isaaclab.devices`
  keyboard, gamepad or SpaceMouse config into the equivalent pipeline and picks the pipeline for a
  script's ``--teleop_device`` request, or ``None`` when ``isaacteleop`` is not installed.
* Added :func:`~isaaclab_teleop.teleop_input.create_teleop_input`, a keyboard, gamepad or
  SpaceMouse that a script drives frame by frame with key callbacks, running through Isaac Capture
  when ``isaacteleop`` is installed and through the deprecated :mod:`isaaclab.devices` device
  otherwise.
* Added :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller`,
  :class:`~isaaclab_teleop.control_pollers.SpaceMouseResetPoller` and
  :func:`~isaaclab_teleop.control_pollers.create_control_pollers`, which turn the ``B`` / ``P`` /
  ``R`` keys and the SpaceMouse's right button into session start, pause and reset.
* Added the ``key_event_sources`` argument to :func:`~isaaclab_teleop.create_isaac_teleop_device`
  and :class:`~isaaclab_teleop.IsaacTeleopDevice`, the input surfaces that feed the pipeline's
  keyboards. It defaults to the ``key_event_source`` of every running visualizer.
* Added :attr:`~isaaclab_teleop.IsaacTeleopDevice.last_step_result`, the most recent full pipeline
  output, for devices whose pipeline declares outputs besides ``"action"``.
* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.drain_pressed_keys`, the keys pressed on the
  session's input surfaces since the last call, each returned once.
