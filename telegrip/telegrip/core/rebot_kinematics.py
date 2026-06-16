"""
reBot kinematics adapter — thin wrapper around the OFFICIAL reBot kinematics.

This module deliberately contains NO custom kinematics math. It calls the
manufacturer's Pinocchio-based solver (``reBotArm_control_py.kinematics``,
the reference implementation aligned with reBot's C++ code) so the arm is
driven by the same forward/inverse kinematics the vendor ships.

Model: the official 6-DOF arm (joint1..joint6). The gripper (motor 7) is NOT
part of this model — it stays a separate motor command, exactly as before.

Telegrip's PyBullet visualizer keeps using the 7-joint URDF (with the gripper
joint) for display; only the IK/FK engine for reBot comes from here.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as _R, Slerp as _Slerp

from ..utils import get_absolute_path

logger = logging.getLogger(__name__)

# 6-joint official URDF (no gripper) used for the Pinocchio IK/FK model.
# Resolved relative to the project root, the same way config resolves URDFs.
_DEFAULT_IK_URDF_REL = "URDF/reBot_DevArm/rebot_devarm_ik.urdf"


class RebotKinematics:
    """
    Pose-based FK/IK for the reBot 6-DOF arm, backed by the official solver.

    All angles are in RADIANS and follow the official URDF joint order
    (joint1..joint6 == shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_yaw, wrist_roll).
    """

    def __init__(self, urdf_path: Optional[str] = None) -> None:
        # Imported here so telegrip still starts if pinocchio is unavailable
        # (only reBot mode needs it). The kinematics package is vendored from the
        # official reBot library — see telegrip/vendor/rebot_kinematics/NOTICE.md.
        from ..vendor.rebot_kinematics import (
            load_robot_model,
            compute_fk,
            get_end_effector_frame_id,
            pos_rot_to_se3,
        )
        from ..vendor.rebot_kinematics.inverse_kinematics import (
            solve_ik,
            IKParams,
        )

        self._compute_fk = compute_fk
        self._solve_ik = solve_ik
        self._pos_rot_to_se3 = pos_rot_to_se3

        path = str(urdf_path or get_absolute_path(_DEFAULT_IK_URDF_REL))
        self._model = load_robot_model(path)
        self._data = self._model.createData()
        self._grav_data = self._model.createData()  # separate buffer for gravity
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self.num_joints = int(self._model.nq)
        # Default solver params (match the vendor's ArmEndPos move_to_ik).
        self._ik_params = IKParams(
            max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6,
        )
        logger.info(
            f"RebotKinematics loaded official model from {path} "
            f"(nq={self.num_joints})"
        )

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Generalized gravity torque (N·m) for joints 1-6 at configuration q.

        Uses Pinocchio's ``computeGeneralizedGravity`` on the official URDF's
        inertial model — the feedforward term for MIT-mode impedance so the arm
        holds against gravity with zero steady-state error and no sag.
        """
        q = np.asarray(q, dtype=float).reshape(self.num_joints)
        return np.asarray(pin.computeGeneralizedGravity(self._model, self._grav_data, q)).copy()

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics.

        Args:
            q: (6,) joint angles in radians.
        Returns:
            (position (3,), rotation (3,3)) of the end-effector in the base frame.
        """
        q = np.asarray(q, dtype=float).reshape(self.num_joints)
        pos, rot, _ = self._compute_fk(self._model, q)
        return pos, rot

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Inverse kinematics for a full 6-DOF pose.

        Args:
            target_pos: (3,) target end-effector position (metres).
            target_rot: (3,3) target end-effector rotation matrix.
            q_seed:     (6,) seed joint angles in radians (e.g. current pose).
        Returns:
            (q (6,), success, error) — q is the solved joint vector in radians.
        """
        q_seed = np.asarray(q_seed, dtype=float).reshape(self.num_joints)
        target = self._pos_rot_to_se3(
            np.asarray(target_pos, dtype=float), np.asarray(target_rot, dtype=float)
        )
        result = self._solve_ik(
            self._model, self._data, self._end_frame_id,
            target, q_seed.copy(), self._ik_params,
        )
        return result.q, bool(result.success), float(result.error)

    @staticmethod
    def step_pose(pos_a: np.ndarray, rot_a: np.ndarray,
                  pos_b: np.ndarray, rot_b: np.ndarray,
                  max_lin: float, max_ang: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advance pose A toward pose B by a bounded step (trajectory servoing).

        Position is linearly interpolated and orientation is slerp'd by the same
        fraction ``u``, where ``u`` is capped so the step is at most ``max_lin``
        metres and ``max_ang`` radians. With A = the last commanded pose and
        B = the live target, repeated calls make the end-effector trace a smooth
        continuous path toward the hand instead of snapping to each target.

        Returns (position (3,), rotation (3x3)).
        """
        pos_a = np.asarray(pos_a, dtype=float)
        pos_b = np.asarray(pos_b, dtype=float)
        rot_a = np.asarray(rot_a, dtype=float)
        rot_b = np.asarray(rot_b, dtype=float)

        dp = pos_b - pos_a
        lin = float(np.linalg.norm(dp))
        rrel = rot_a.T @ rot_b
        ang = float(np.arccos(np.clip((np.trace(rrel) - 1.0) / 2.0, -1.0, 1.0)))

        u = 1.0
        if max_lin > 0.0 and lin > max_lin:
            u = min(u, max_lin / lin)
        if max_ang > 0.0 and ang > max_ang:
            u = min(u, max_ang / ang)

        pos = pos_a + u * dp
        if ang < 1e-9:
            rot = rot_b
        else:
            slerp = _Slerp([0.0, 1.0], _R.from_matrix([rot_a, rot_b]))
            rot = slerp([u]).as_matrix()[0]
        return pos, rot
