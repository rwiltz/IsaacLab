Removed
^^^^^^^

* Removed the ``/isaaclab/xr/auto_start`` carb setting. It conflated visualizer pumping with XR
  session start; the former now follows ``/isaaclab/xr/enabled``, and the latter moved to
  :mod:`isaaclab_teleop`, where it is gated on CloudXR runtime ownership.
