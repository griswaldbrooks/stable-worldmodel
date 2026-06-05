"""Scripted IK oracle for SO-101 pick-and-lift.

Ported from the picknik ``end-to-end`` repo's ``OraclePickLift`` and
restructured for stable-worldmodel: the original drove the Genesis scene
itself (``scene.step()`` inside its move loop); here ``World`` owns the
stepping, so the oracle *plans* per-phase IK joint targets at episode
start and *emits* one normalized action per control step.

The grasp is the de-risked open-ENCLOSE-close recipe (ADR-0005 /
genesis-nyx-derisk): full torque swings the jaw open to a moderate angle,
the cube enters the open mouth on descend, and a firmer force-limited
close swings the jaw shut on it. The per-phase gripper force limits are
what make this work — the default ±3.35 N·m crushes *through* the cube —
and they are not expressible through the env's action space, so the
policy applies them directly to the arm at phase transitions (privileged
access, same spirit as PushT's ``WeakPolicy`` reaching into pymunk).

The TCP is the moving-jaw collision AABB centre in the gripper link's
local frame — NOT the ``gripperframe`` MJCF site, which is the wrist-cam
look-at ~10 cm past the fingers. A 5-DoF arm cannot satisfy an arbitrary
6-DoF pose but can reach a vertical fingers-down grasp: the target
orientation is ``Ry(+90°) * home_quat``.
"""

import numpy as np

from stable_worldmodel.policy import BasePolicy

from .env import JOINT_CTRL_HIGH, JOINT_CTRL_LOW


# Grasp recipe constants (verified on the CPU backend in end-to-end).
GRIPPER_DOF_INDEX = 5
GRIPPER_CLOSED = -0.15
# Open to a *moderate* angle, not the full 1.7 — full open is an
# unnecessarily long arc the close then can't cover in time.
GRIPPER_OPEN_GRASP = 0.85
# Torque while opening the jaw (the MJCF sts3215 forcerange); the close
# force-cap below is too small to swing the jaw open at all.
GRIPPER_OPEN_FORCE = 2.94
# Enough torque to swing the jaw onto the cube and grip without crushing
# through it.
GRIPPER_CLOSE_FORCE = 0.6
# The IK TCP is the moving-jaw AABB centre, but the pinch point sits
# ~half a finger-gap toward the base along the approach axis. Without
# this shift the open jaws straddle the cube yet the closing jaw pinches
# past it. World-frame offset, calibrated for the top-down grasp.
GRASP_CENTRE_OFFSET = 0.018

_GRIPPER_LINK = 'gripper'
_JAW_LINK = 'moving_jaw_so101_v1'

# Phase durations in *control* steps (at the env's 25 Hz): approach,
# descend, close, lift. Same wall-clock as the end-to-end oracle's
# (140, 110, 90, 200) sim steps at its dt=0.01 baseline.
DEFAULT_PHASE_STEPS = (35, 28, 23, 50)


# --- quaternion helpers (wxyz, scalar-first — Genesis convention) --------


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _axis_angle(axis, deg):
    half = np.deg2rad(deg) / 2.0
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    return np.array([np.cos(half), *(a * np.sin(half))])


def _quat_conj_rotate(q, v):
    """Rotate ``v`` by the conjugate of unit quat ``q`` (world -> local)."""
    w, x, y, z = q
    u = np.array([-x, -y, -z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _to_np(x):
    return np.asarray(x.cpu() if hasattr(x, 'cpu') else x, dtype=np.float64)


def _normalize(q):
    """Joint targets (rad) -> action in [-1, 1] per the env's ctrlrange."""
    a = 2.0 * (q - JOINT_CTRL_LOW) / (JOINT_CTRL_HIGH - JOINT_CTRL_LOW) - 1.0
    return np.clip(a, -1.0, 1.0).astype(np.float32)


class _Phase:
    __slots__ = ('z', 'gripper', 'steps', 'gripper_force')

    def __init__(self, z, gripper, steps, gripper_force):
        self.z = z
        self.gripper = gripper
        self.steps = steps
        self.gripper_force = gripper_force


class _Plan:
    """A per-episode phase schedule with lazy per-phase IK.

    Each phase's IK is solved when the phase *starts*, warm-started from
    the arm's current configuration — Genesis IK is damped least squares
    seeded at the current pose, so solving the lift target from the grasp
    pose (rather than everything up front from home) keeps solutions on
    the same branch. This matches the original oracle's ``_move_to``,
    which re-solved at every phase boundary. Holds the last action once
    the schedule is exhausted.

    """

    def __init__(self, phases, solve):
        self.phases = phases
        self.solve = solve
        self.idx = -1
        self.remaining = 0
        self.action = None
        self.pending_force = None

    def next_action(self):
        if self.remaining == 0 and self.idx < len(self.phases) - 1:
            self.idx += 1
            phase = self.phases[self.idx]
            self.action = self.solve(phase.z, phase.gripper)
            self.remaining = phase.steps
            self.pending_force = phase.gripper_force
        if self.remaining > 0:
            self.remaining -= 1
        force = self.pending_force
        self.pending_force = None
        return self.action, force


class OraclePickLiftPolicy(BasePolicy):
    """Deterministic 4-phase pick-and-lift expert for ``SO101PickCube``.

    Plans at episode start (detected via ``info['step_idx'] == 0``) by
    reading the cube position from the env and solving IK for the four
    phase targets; then plays the schedule back one action per step.
    """

    def __init__(
        self,
        approach_z=0.10,
        grasp_z=0.02,
        lift_z=0.18,
        phase_steps=DEFAULT_PHASE_STEPS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.approach_z = approach_z
        self.grasp_z = grasp_z
        self.lift_z = lift_z
        self.phase_steps = tuple(phase_steps)
        self._plans = {}

    def set_env(self, env):
        self.env = env
        spec = getattr(env, 'spec', None)
        if spec is None:
            envs = getattr(env, 'envs', None)
            if envs and len(envs) > 0:
                spec = envs[0].spec
        assert spec is not None and 'swm/SO101' in spec.id, (
            'OraclePickLiftPolicy can only be used with SO-101 envs.'
        )

    def _unwrapped_envs(self):
        if hasattr(self.env, 'envs'):
            return [e.unwrapped for e in self.env.envs]
        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            return [e.unwrapped for e in base_env.envs]
        return [base_env]

    def _plan(self, env):
        """Solve the 4 phase IK targets for this env's current cube pose."""
        arm = env._arm
        glink = arm.get_link(_GRIPPER_LINK)
        jaw = arm.get_link(_JAW_LINK)

        # TCP: moving-jaw collision AABB centre in the gripper link's
        # local frame, calibrated at the (home) pose the env resets to.
        jaw_aabb = _to_np(jaw.get_AABB()).reshape(-1, 3)
        jaw_centre = jaw_aabb.mean(axis=0)
        gpos = _to_np(glink.get_pos())
        gquat = _to_np(glink.get_quat())
        tcp_local = _quat_conj_rotate(gquat, jaw_centre - gpos)

        grasp_xy = _to_np(env._cube.get_pos())[:2]
        # Top-down: rotate home so the finger axis points at world -Z.
        target_quat = _qmul(_axis_angle((0, 1, 0), 90.0), gquat)
        # Shift the IK target toward the base so the grasp CENTRE
        # (between the fingers) lands on the cube.
        r = float(np.hypot(*grasp_xy))
        if r > 1e-6:
            grasp_xy = grasp_xy - GRASP_CENTRE_OFFSET * grasp_xy / r

        def solve(z, gripper):
            q = _to_np(
                arm.inverse_kinematics(
                    link=glink,
                    pos=np.array([grasp_xy[0], grasp_xy[1], z]),
                    quat=target_quat,
                    local_point=tcp_local,
                    rot_mask=[True, True, True],
                    pos_mask=[True, True, True],
                )
            )
            q[GRIPPER_DOF_INDEX] = gripper
            return _normalize(q)

        s_app, s_desc, s_close, s_lift = self.phase_steps
        return _Plan(
            [
                _Phase(
                    self.approach_z,
                    GRIPPER_OPEN_GRASP,
                    s_app,
                    GRIPPER_OPEN_FORCE,
                ),
                _Phase(self.grasp_z, GRIPPER_OPEN_GRASP, s_desc, None),
                _Phase(
                    self.grasp_z,
                    GRIPPER_CLOSED,
                    s_close,
                    GRIPPER_CLOSE_FORCE,
                ),
                _Phase(self.lift_z, GRIPPER_CLOSED, s_lift, None),
            ],
            solve,
        )

    @staticmethod
    def _set_gripper_force(env, f):
        env._arm.set_dofs_force_range(
            np.array([-f]), np.array([f]), dofs_idx_local=[GRIPPER_DOF_INDEX]
        )

    def get_action(self, info_dict, **kwargs):
        assert self.env is not None, 'Environment not set for the policy'

        envs = self._unwrapped_envs()
        step_idx = np.asarray(info_dict['step_idx'])
        if step_idx.ndim > 1:  # (envs, history) -> latest
            step_idx = step_idx[:, -1]
        step_idx = np.atleast_1d(step_idx)

        actions = np.zeros((len(envs), len(JOINT_CTRL_LOW)), dtype=np.float32)
        for i, env in enumerate(envs):
            if i not in self._plans or step_idx[i] == 0:
                self._plans[i] = self._plan(env)
            action, force = self._plans[i].next_action()
            if force is not None:
                self._set_gripper_force(env, force)
            actions[i] = action

        if not hasattr(self.env, 'envs') and not hasattr(
            self.env.unwrapped, 'envs'
        ):
            return actions[0]
        return actions
