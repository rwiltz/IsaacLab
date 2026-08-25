Changed
^^^^^^^

* **Breaking:** The removed ``Se3SpaceMouseCfg``-based ``teleop_devices`` entries on the
  ``contrib`` place/stack environment configs are gone. Spacemouse teleoperation for these
  environments is still available by passing ``--teleop_device spacemouse`` to
  ``teleop_se3_agent.py`` / ``record_demos.py``, which now resolves an
  :class:`~isaaclab_teleop.IsaacTeleopCfg` via :func:`~isaaclab_teleop.se3_spacemouse_teleop_cfg`
  independent of what the environment config declares.
