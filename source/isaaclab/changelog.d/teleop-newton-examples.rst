Fixed
^^^^^

* Fixed the Pink IK action term resolving its hand joints in articulation order. The action tensor
  carries hand targets in the order ``hand_joint_names`` declares, so the resolved ids have to keep
  that order; without ``preserve_order`` each target is applied to whichever joint the articulation
  lists in that slot. PhysX orders these robots' joints the way the config lists them, so the
  mismatch is invisible there, but Newton groups them per finger and most hand targets end up
  driving another joint, including past their own limits.
