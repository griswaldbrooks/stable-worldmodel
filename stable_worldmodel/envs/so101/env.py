"""SO-101 pick-cube environment in the Genesis simulator.

Loads the vendored SO-101 MJCF (``sim/so101_new_calib.xml``, the LeRobot
"new calibration" with the virtual zero at mid-range) into a Genesis scene
with a ground plane and a small cube the arm can pick. Ported from the
``GenesisTeleopIO`` adapter and ``oracle_policy`` grasp recipe in the
picknik ``end-to-end`` repo, restructured as a ``gym.Env``.

Physics follows the de-risked grasp recipe: friction on both the arm and
the cube, the multi-contact rigid solver, a finite CoACD decomposition
threshold so the gripper mouth stays open (Genesis' robot default is a
single convex hull per mesh, which fills the mouth solid), and a small
``dt`` so contact is stiff enough to hold a clamp. Without these the
gripper can never hold the cube.

Actions are absolute joint-position targets in ``[-1, 1]``, denormalized
to the MJCF actuator ``ctrlrange`` (matching the LeRobot SO-101 joint
order). Success is the cube lifted ``LIFT_THRESHOLD`` above its rest
height without flying off (a stability check, ported from the oracle).

Heavy: requires the ``genesis`` extra (``uv sync --extra genesis``). The
Genesis backend is fixed at the process' first ``gs.init()`` — this env
initializes the CPU backend (rasterizer path, CI-safe).
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces


ASSETS_DIR = Path(__file__).resolve().parent / 'sim'
MJCF_PATH = ASSETS_DIR / 'so101_new_calib.xml'

# LeRobot SO-101 joint order; produced data is shape-compatible with
# policies trained on the community SO-101 datasets.
SO101_JOINTS = (
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
)

# Actuator ctrlrange from sim/so101_new_calib.xml (radians). Actions in
# [-1, 1] denormalize linearly into these per-joint ranges.
JOINT_CTRL_LOW = np.array(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453],
    dtype=np.float32,
)
JOINT_CTRL_HIGH = np.array(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
    dtype=np.float32,
)

# --- grasp-capable physics recipe (ported from oracle_policy.py) ---------
# Friction in [1.5, 2.0] lifts reliably; >= ~2.5 ejects the cube.
GRASP_FRICTION = 1.8
# Genesis' robot default is inf (one convex hull per mesh -> the gripper
# mouth fills in solid and the jaws can never enclose anything). A finite
# threshold makes CoACD decompose the fingers so the mouth re-opens.
DECOMPOSE_ROBOT_ERROR_THRESHOLD = 0.05
# Smaller dt => lower 2*dt floor on the contact time-constant => stiffer
# contact => less penetration during a grasp.
SIM_DT = 0.005
# One env step = CONTROL_HZ; with SIM_DT this is an integer substep count.
CONTROL_HZ = 25

# Cube lifted this far above rest height (m) counts as success.
LIFT_THRESHOLD = 0.05

_GRIPPER_LINK_NAME = 'gripper'

# The `gripperframe` MJCF site (canonical tool-center-point) in the
# gripper link's local frame — the wrist camera's look-at direction.
_GRIPPERFRAME_POS = np.array(
    [-0.0079, -0.000218121, -0.0981274], dtype=np.float64
)

# Genesis' default near plane (10 cm) clips wrist cameras sitting a few
# cm from the gripper geometry. Override to 1 mm.
_CAMERA_NEAR_M = 0.001

# Workspace overview: 3/4 side angle framing the arm and the cube.
_SCENE_CAMERA = {
    'pos': (0.40, 0.20, 0.15),
    'lookat': (0.15, 0.0, 0.05),
}

_CUBE_SIZE = (0.03, 0.03, 0.03)
_CUBE_POS = (0.15, 0.0, 0.015)
# Pale sage green: chroma + shading contrast so edges read at low
# resolution (default white renders nearly flat).
_CUBE_COLOR = (0.45, 0.70, 0.45, 1.0)
# Goal image: cube teleported this far above rest, arm at home — a
# deterministic "lifted" still life (same trick as end-to-end's
# cube_lifted scene state).
_GOAL_LIFT_DZ = 0.085

DEFAULT_VARIATIONS = ('cube.start_position',)

# --- Nyx photoreal renderer (renderer='nyx') ------------------------------
# Nyx is Genesis' GPU path tracer, wired in as camera *sensors*. It renders
# the same scene photorealistically, lit by an HDRI environment map.
# Requires the gs-nyx + gs-nyx-plugin wheels (genesis extra) and an NVIDIA
# GPU (driver 575+, CUDA 12.9+).
RENDERERS = ('rasterizer', 'nyx')
_NYX_WRIST_FOV = 60.0
_NYX_SCENE_FOV = 40.0
_NYX_SPP = 48
_NYX_ENV_MULTIPLIER = 4.0


def _ensure_genesis_initialized(backend='cpu'):
    """Initialize Genesis once per process. Idempotent.

    Genesis raises if init'd twice, and the backend is fixed at the first
    ``gs.init()`` — defer to Genesis' own global state so several envs
    (or env + other Genesis users) can share a process. A backend
    mismatch (e.g. a Nyx env after a CPU env already init'd Genesis) is
    unrecoverable in-process, so fail loudly.
    """
    import genesis as gs

    gs_backend = gs.gpu if backend == 'gpu' else gs.cpu
    if not gs._initialized:
        gs.init(backend=gs_backend, logging_level='warning')
        return
    if backend == 'gpu' and gs.backend == gs.cpu:
        raise RuntimeError(
            'Genesis is already initialized on the CPU backend in this '
            "process, but renderer='nyx' needs the GPU backend. The "
            'backend is fixed at the first gs.init() — construct the Nyx '
            'env in a fresh process (e.g. a subprocess).'
        )


def _look_at(eye, target, up_hint):
    """Standard look-at 4x4 (camera looks down -Z, +Y is image-up)."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = -forward
    T[:3, 3] = eye
    return T


def _wrist_camera_offset(side_x=0.055):
    """Camera pose in the gripper link's local frame: laterally off the
    wrist-roll housing, looking past the gripperframe along the gripper's
    reach axis (jaws in the foreground, workspace filling the FOV)."""
    eye = np.array([side_x, 0.0, 0.0], dtype=np.float64)
    direction = _GRIPPERFRAME_POS / np.linalg.norm(_GRIPPERFRAME_POS)
    target = direction * 0.30
    return _look_at(eye, target, np.array([1.0, 0.0, 0.0]))


def _to_np(x, dtype=np.float64):
    return np.asarray(x.cpu() if hasattr(x, 'cpu') else x, dtype=dtype)


def _yaw_quat(yaw):
    """wxyz (scalar-first, Genesis convention) quat for a world-Z yaw."""
    half = float(yaw) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class SO101PickCube(gym.Env):
    """Pick a cube off the table with a Genesis-simulated SO-101 arm."""

    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': CONTROL_HZ,
        'video.frames_per_second': CONTROL_HZ,
    }

    def __init__(
        self,
        resolution=224,
        render_mode='rgb_array',
        show_viewer=False,
        settle_steps=80,
        multiview=False,
        renderer='rasterizer',
        nyx_env_map=None,
        init_value=None,
    ):
        if renderer not in RENDERERS:
            raise ValueError(
                f'unknown renderer {renderer!r}; valid: {RENDERERS}'
            )
        if renderer == 'nyx' and nyx_env_map is None:
            raise ValueError(
                "renderer='nyx' requires nyx_env_map (an HDRI .hdr/.exr)"
            )
        # Nyx is GPU-only; the rasterizer stays on CPU (CI-safe).
        _ensure_genesis_initialized('gpu' if renderer == 'nyx' else 'cpu')
        import genesis as gs

        self.render_mode = render_mode
        self.render_size = resolution
        self._settle_steps = settle_steps
        self._multiview = multiview
        self._renderer = renderer
        self._nyx_env_map = nyx_env_map
        self._substeps = round(1.0 / (SIM_DT * CONTROL_HZ))

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=SIM_DT),
            rigid_options=gs.options.RigidOptions(
                # Genesis 1.0 non-convex contact; required to hold a clamp.
                enable_multi_contact=True,
                box_box_detection=True,
                iterations=200,
                ls_iterations=100,
                noslip_iterations=8,
            ),
            show_viewer=show_viewer,
        )
        material = gs.materials.Rigid(friction=GRASP_FRICTION)
        self._arm = self._scene.add_entity(
            gs.morphs.MJCF(
                file=str(MJCF_PATH),
                decompose_robot_error_threshold=(
                    DECOMPOSE_ROBOT_ERROR_THRESHOLD
                ),
            ),
            material=material,
        )
        # The ground gets the grasp material too: the original setup
        # loaded scene.xml, whose MJCF floor inherits the arm entity's
        # friction — high cube-ground friction keeps the cube from
        # scooting away from the closing jaw during the enclose grasp.
        self._ground = self._scene.add_entity(
            gs.morphs.Plane(), material=material
        )
        self._cube = self._scene.add_entity(
            gs.morphs.Box(size=_CUBE_SIZE, pos=_CUBE_POS),
            material=material,
            surface=gs.surfaces.Default(color=_CUBE_COLOR),
        )

        # Cameras: a fixed workspace overview + a wrist camera attached
        # to the gripper link. Rasterizer cameras are re-posed per render
        # via move_to_attach(); Nyx sensors attach natively via offset_T
        # at construction. Genesis/Nyx `res` is (width, height).
        gripper_link = next(
            link for link in self._arm.links if link.name == _GRIPPER_LINK_NAME
        )
        if renderer == 'nyx':
            self._add_nyx_cameras(resolution, gripper_link)
            self._scene.build()
        else:
            self._scene_cam = self._scene.add_camera(
                res=(resolution, resolution),
                pos=_SCENE_CAMERA['pos'],
                lookat=_SCENE_CAMERA['lookat'],
            )
            self._wrist_cam = self._scene.add_camera(
                res=(resolution, resolution),
                pos=_SCENE_CAMERA['pos'],
                lookat=_SCENE_CAMERA['lookat'],
                near=_CAMERA_NEAR_M,
            )
            self._scene.build()
            # After build, attach the wrist camera to the gripper link;
            # it is re-posed each render via move_to_attach().
            self._wrist_cam.attach(gripper_link, _wrist_camera_offset())

        # MJCF home pose (new calibration: all zeros, mid-range).
        self._home_dofs = _to_np(self._arm.get_dofs_position(), np.float32)

        n = len(SO101_JOINTS)
        # proprio: joint positions (rad). state: proprio + cube pose
        # (pos xyz + quat wxyz). Transient contact can push joints
        # slightly past their MJCF range, so bounds stay infinite.
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32
                ),
                'state': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n + 7,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n,), dtype=np.float32
        )

        self.variation_space = swm_spaces.Dict(
            {
                'cube': swm_spaces.Dict(
                    {
                        # Bounds are the verified basin of the scripted
                        # oracle's grasp (~70% success inside; see
                        # expert_policy.py). The grasp recipe was only
                        # ever validated near (0.15, 0) upstream; widening
                        # this window is tracked follow-up work.
                        'start_position': swm_spaces.Box(
                            low=np.array([0.14, -0.02]),
                            high=np.array([0.15, 0.02]),
                            init_value=np.array(
                                _CUBE_POS[:2], dtype=np.float64
                            ),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        'angle': swm_spaces.Box(
                            low=-np.pi,
                            high=np.pi,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                    }
                ),
            }
        )
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        self._cube_rest_z = _CUBE_POS[2]
        self._goal = None
        self.env_name = 'SO101PickCube'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        self._scene.reset()
        xy = self.variation_space['cube']['start_position'].value
        yaw = self.variation_space['cube']['angle'].value
        self._cube.set_pos(np.array([xy[0], xy[1], _CUBE_SIZE[2] / 2.0]))
        self._cube.set_quat(_yaw_quat(yaw))

        # Settle: hold home while the cube comes to rest on the plane.
        for _ in range(self._settle_steps):
            self._arm.control_dofs_position(self._home_dofs)
            self._scene.step()
        self._cube_rest_z = float(_to_np(self._cube.get_pos())[2])

        self._goal = self._render_goal()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        target = JOINT_CTRL_LOW + (np.clip(action, -1.0, 1.0) + 1.0) * (
            0.5 * (JOINT_CTRL_HIGH - JOINT_CTRL_LOW)
        )
        self._arm.control_dofs_position(target)
        for _ in range(self._substeps):
            self._scene.step()

        obs = self._get_obs()
        info = self._get_info()

        cube_pos = _to_np(self._cube.get_pos())
        dz = float(cube_pos[2]) - self._cube_rest_z
        # Stability gate from the oracle: a cube ejected by an unstable
        # contact does not count as lifted.
        stable = abs(cube_pos[2]) < 5.0 and bool(
            np.all(np.abs(cube_pos[:2]) < 2.0)
        )
        success = dz > LIFT_THRESHOLD and stable
        reward = dz  # the higher the better

        terminated = success
        truncated = False
        return obs, reward, terminated, truncated, info

    def _add_nyx_cameras(self, resolution, gripper_link):
        """Build the Nyx photoreal cameras as sensors on the scene.

        The wrist camera attaches natively to the gripper link via
        ``offset_T`` (the same 4x4 local-frame pose the rasterizer path
        passes to attach()); the scene camera is static. Both are lit by
        the HDRI env map. Nyx ``read().rgb`` returns (height, width, 3).
        """
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as nps
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

        env_map = nps.EnvironmentMapAsset()
        env_map.texture = str(self._nyx_env_map)
        env_map.layout = nps.EEnvMapLayout.LongLat
        env_map.multiplier = _NYX_ENV_MULTIPLIER
        common = dict(
            spp=_NYX_SPP,
            denoise=True,
            render_mode=npr.ERenderMode.FastPathTracer,
            env_maps=[env_map],
        )
        self._wrist_cam = self._scene.add_sensor(
            NyxCameraOptions(
                res=(resolution, resolution),
                fov=_NYX_WRIST_FOV,
                near=_CAMERA_NEAR_M,
                entity_idx=self._arm.idx,
                link_idx_local=gripper_link.idx_local,
                offset_T=_wrist_camera_offset(),
                **common,
            )
        )
        self._scene_cam = self._scene.add_sensor(
            NyxCameraOptions(
                res=(resolution, resolution),
                fov=_NYX_SCENE_FOV,
                pos=_SCENE_CAMERA['pos'],
                lookat=_SCENE_CAMERA['lookat'],
                **common,
            )
        )

    def _read_camera(self, cam):
        """Renderer-agnostic single-camera read -> (h, w, 3) uint8."""
        if self._renderer == 'nyx':
            rgb = cam.read().rgb
            img = np.asarray(rgb.cpu() if hasattr(rgb, 'cpu') else rgb)
        else:
            # force_render: the visualizer skips state sync when scene._t
            # is unchanged, which makes it blind to set_pos() teleports
            # (goal rendering) and to anything between reset() and the
            # first step.
            img, *_ = cam.render(rgb=True, force_render=True)
        return np.ascontiguousarray(np.asarray(img)[..., :3], dtype=np.uint8)

    def render(self):
        return self._read_camera(self._scene_cam)

    def render_multiview(self):
        """Both cameras as a dict; the pixels wrapper turns these into
        ``pixels.scene`` / ``pixels.wrist`` info keys.

        Opt-in via ``multiview=True`` (same pattern as the ogbench envs):
        with it off this falls back to the single scene camera, so the
        plain ``pixels`` info key — which video recording expects — keeps
        working.
        """
        if not self._multiview:
            return self.render()
        if self._renderer != 'nyx':
            self._wrist_cam.move_to_attach()
        return {
            'scene': self.render(),
            'wrist': self._read_camera(self._wrist_cam),
        }

    def _render_goal(self):
        """Goal image: cube teleported to its lifted height, arm at home.

        Rasterizer: no physics step happens between teleport and restore
        (force_render makes the teleport visible), so the dynamics are
        unaffected. Nyx: sensors only refresh on scene.step(), so the
        teleport and the restore are each followed by a single step (the
        arm holds home; 5 ms of cube free-fall is ~0.1 mm) and the cube
        state is restored exactly afterwards.
        """
        pos = _to_np(self._cube.get_pos())
        quat = _to_np(self._cube.get_quat())
        lifted = pos.copy()
        lifted[2] = self._cube_rest_z + _GOAL_LIFT_DZ
        self._cube.set_pos(lifted)
        if self._renderer == 'nyx':
            self._scene.step()
        goal = self.render()
        self._cube.set_pos(pos)
        self._cube.set_quat(quat)
        if self._renderer == 'nyx':
            try:
                self._cube.set_dofs_velocity(np.zeros(6))
            except Exception:
                pass
            self._scene.step()
        return goal

    def _get_obs(self):
        qpos = _to_np(self._arm.get_dofs_position(), np.float32)
        cube_pos = _to_np(self._cube.get_pos(), np.float32)
        cube_quat = _to_np(self._cube.get_quat(), np.float32)
        state = np.concatenate([qpos, cube_pos, cube_quat])
        return {'proprio': qpos, 'state': state}

    def _get_info(self):
        cube_pos = _to_np(self._cube.get_pos())
        return {
            'env_name': self.env_name,
            'cube_pos': cube_pos,
            'cube_rest_z': self._cube_rest_z,
            'goal': self._goal,
        }

    def close(self):
        # Genesis scenes don't expose an explicit teardown; drop
        # references so GC can reclaim. Idempotent.
        self._scene_cam = None
        self._wrist_cam = None
        self._cube = None
        self._arm = None
        self._scene = None
