* Changed the default ``dead_zone`` of :class:`~isaaclab.devices.Se2GamepadCfg` and
  :class:`~isaaclab.devices.Se3GamepadCfg` from 0.01 to 0.15. An idle gamepad's stick drift
  exceeds 0.01, so the old default moved the robot with nobody touching the pad.
