Changed
^^^^^^^

* **Breaking:** Changed XR session start to follow CloudXR runtime ownership instead of
  headlessness. The OpenXR/AR session now starts automatically whenever Isaac Lab launches the
  CloudXR runtime, including under ``--visualizer kit``, so **Start XR** no longer has to be
  clicked. Isaac Lab leaves session start to the operator whenever it does not launch the
  runtime: with ``--no-auto_launch_cloudxr``, ``ISAACLAB_CXR_SKIP_AUTOLAUNCH=1``, or
  ``--cloudxr_env none``.
