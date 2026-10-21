# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limit wrist target speeds between absolute G1 commands."""

import math

import torch


class WristTrajectory:
    """Advance both wrists from their last commanded poses with bounded speeds.

    Actions have shape (1, 28): two XYZ + XYZW wrist poses followed by fourteen
    finger targets. Wrist positions use meters in the environment frame. Finger
    targets pass through unchanged. This limits velocity, not acceleration or jerk.
    """

    def __init__(self, initial_action: torch.Tensor, linear_speed: float = 0.025, angular_speed: float = 0.4):
        """Initialize from an action with speeds in [m/s] and [rad/s]."""
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (linear_speed, angular_speed)
        ):
            raise ValueError("Wrist speeds must be positive and finite.")
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.reset(initial_action)

    def reset(self, action: torch.Tensor) -> None:
        """Reanchor to a measured or explicitly held pose after a reset or stop."""
        self._wrists = self._validated_action(action)[0, :14].reshape(2, 7).clone()

    def advance(self, target_action: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the next bounded wrist poses and the target's unchanged fingers.

        New targets start from the previous output, including during retargeting.
        The input and returned tensors do not alias internal trajectory state.

        Args:
            target_action: Absolute G1 action with shape (1, 28).
            dt: Positive finite control timestep [s].
        """
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Wrist timestep must be positive and finite.")
        if target_action.device != self._wrists.device or target_action.dtype != self._wrists.dtype:
            raise ValueError("Wrist targets must retain the initial action's device and dtype.")
        result = self._validated_action(target_action)
        target = result[0, :14].reshape(2, 7)
        delta = target[:, :3] - self._wrists[:, :3]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        if not torch.isfinite(distance).all():
            raise ValueError("Wrist position differences must have finite norms.")
        position = self._wrists[:, :3] + delta * (self.linear_speed * dt / distance.clamp_min(1e-12)).clamp(max=1)

        current_quat = self._wrists[:, 3:]
        target_quat = target[:, 3:]
        dot = (current_quat * target_quat).sum(dim=-1, keepdim=True)
        target_quat = torch.where(dot < 0, -target_quat, target_quat)
        # Chord lengths retain accuracy for small rotations where acos(dot)
        # loses precision. Quaternions cover half the physical rotation angle.
        half_angle = 2 * torch.atan2(
            torch.linalg.vector_norm(target_quat - current_quat, dim=-1, keepdim=True),
            torch.linalg.vector_norm(target_quat + current_quat, dim=-1, keepdim=True),
        )
        fraction = (self.angular_speed * dt / (2 * half_angle).clamp_min(1e-12)).clamp(max=1)
        quaternion = (
            torch.sin((1 - fraction) * half_angle) * current_quat + torch.sin(fraction * half_angle) * target_quat
        ) / torch.sin(half_angle).clamp_min(1e-12)
        quaternion = torch.where(half_angle < 1e-6, current_quat.lerp(target_quat, fraction), quaternion)
        quaternion /= torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        wrists = torch.cat((position, quaternion), dim=-1)
        if not torch.isfinite(wrists).all():
            raise ValueError("Cannot produce finite wrist targets from this action.")
        self._wrists = wrists
        result[0, :14] = wrists.flatten()
        return result

    @staticmethod
    def _validated_action(action: torch.Tensor) -> torch.Tensor:
        if action.shape != (1, 28) or not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError("Expected a finite floating G1 action with shape (1, 28).")
        result = action.clone()
        wrists = result[0, :14].reshape(2, 7)
        norms = torch.linalg.vector_norm(wrists[:, 3:], dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms < 1e-6).any():
            raise ValueError("Wrist quaternions must have finite, nonzero norms.")
        wrists[:, 3:] /= norms
        return result
