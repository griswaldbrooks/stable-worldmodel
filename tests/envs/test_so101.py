"""Tests for the Genesis-simulated SO-101 pick-cube env and its oracle.

Anything that builds a Genesis scene is marked ``genesis`` (deselected by
default; run locally with ``pytest -m genesis`` after
``uv sync --extra genesis``). Scene construction is expensive, so all
Genesis tests share one module-scoped env.
"""

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers swm/ envs)


# ----------------------------------------------------------------------
# Light tests: no Genesis required.
# ----------------------------------------------------------------------


def test_registered():
    from stable_worldmodel.envs import WORLDS

    assert 'swm/SO101PickCube-v0' in WORLDS
    spec = gym.spec('swm/SO101PickCube-v0')
    assert spec.max_episode_steps == 200


def test_package_imports_without_genesis():
    """The so101 package (env + policy) must import with genesis absent."""
    code = (
        'import sys; sys.modules["genesis"] = None\n'
        'from stable_worldmodel.envs.so101 import ('
        'SO101PickCube, OraclePickLiftPolicy)\n'
        'print("ok")\n'
    )
    out = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert 'ok' in out.stdout


def test_action_normalization_roundtrip():
    from stable_worldmodel.envs.so101.env import (
        JOINT_CTRL_HIGH,
        JOINT_CTRL_LOW,
    )
    from stable_worldmodel.envs.so101.expert_policy import _normalize

    rng = np.random.default_rng(0)
    q = rng.uniform(JOINT_CTRL_LOW, JOINT_CTRL_HIGH)
    a = _normalize(q)
    assert a.shape == q.shape and a.dtype == np.float32
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    # env-side denormalization (mirrors SO101PickCube.step)
    target = JOINT_CTRL_LOW + (a + 1.0) * 0.5 * (
        JOINT_CTRL_HIGH - JOINT_CTRL_LOW
    )
    np.testing.assert_allclose(target, q, atol=1e-5)


# ----------------------------------------------------------------------
# Genesis tests: one shared env, CPU rasterizer backend.
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def env():
    pytest.importorskip('genesis')
    env = gym.make('swm/SO101PickCube-v0')
    yield env
    env.close()


@pytest.mark.genesis
def test_reset_obs_contract(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs['proprio'].shape == (6,)
    assert obs['state'].shape == (13,)
    assert info['env_name'] == 'SO101PickCube'
    assert 'goal' in info and info['goal'].dtype == np.uint8


@pytest.mark.genesis
def test_step_contract(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert 'cube_pos' in info


@pytest.mark.genesis
def test_render(env):
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8
    # default: multiview falls back to the single scene camera so the
    # plain `pixels` key (and video recording) keeps working
    mv = env.unwrapped.render_multiview()
    assert isinstance(mv, np.ndarray)


@pytest.mark.genesis
def test_goal_image_differs_from_scene(env):
    """The goal still-life (lifted cube) must differ from the live view."""
    obs, info = env.reset(seed=0)
    current = env.render()
    diff = np.abs(info['goal'].astype(int) - current.astype(int)).mean()
    assert diff > 1.0


@pytest.mark.genesis
def test_variation_explicit_value(env):
    target = np.array([0.145, 0.01])
    obs, info = env.reset(
        seed=0,
        options={'variation_values': {'cube.start_position': target}},
    )
    np.testing.assert_allclose(info['cube_pos'][:2], target, atol=5e-3)


@pytest.mark.genesis
def test_variation_resamples_across_seeds(env):
    _, info_a = env.reset(seed=0)
    _, info_b = env.reset(seed=1)
    assert not np.allclose(info_a['cube_pos'][:2], info_b['cube_pos'][:2])


@pytest.mark.genesis
@pytest.mark.gpu
def test_nyx_renderer(tmp_path):
    """Nyx path-traced rendering: both cameras, goal still-life.

    Runs in a subprocess for a clean CUDA ``gs.init()`` — the other
    genesis tests initialize Genesis on the CPU backend in-process, and
    the backend is fixed at first init. The HDRI is synthesized so no
    binary asset is committed.
    """
    pytest.importorskip('gs_nyx_plugin')
    iio = pytest.importorskip('imageio.v3')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Nyx requires a CUDA GPU')

    hdr = tmp_path / 'flat.hdr'
    iio.imwrite(hdr, np.full((16, 32, 3), 0.8, dtype=np.float32))

    script = (
        'import numpy as np\n'
        'import gymnasium as gym\n'
        'import stable_worldmodel\n'
        "env = gym.make('swm/SO101PickCube-v0', renderer='nyx',\n"
        f'               nyx_env_map={str(hdr)!r},\n'
        '               multiview=True, resolution=128)\n'
        'obs, info = env.reset(seed=0)\n'
        'scene = env.render()\n'
        'mv = env.unwrapped.render_multiview()\n'
        'assert scene.shape == (128, 128, 3) and scene.dtype == np.uint8\n'
        "assert mv['wrist'].shape == (128, 128, 3)\n"
        'assert scene.mean() > 1.0, "black render"\n'
        "diff = np.abs(info['goal'].astype(int) - scene.astype(int)).mean()\n"
        'assert diff > 1.0, "goal identical to scene"\n'
        'env.close()\n'
        'print("NYX_ENV_OK")\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert 'NYX_ENV_OK' in result.stdout, (
        f'Nyx env run failed.\nSTDOUT:\n{result.stdout[-2000:]}\n'
        f'STDERR:\n{result.stderr[-2000:]}'
    )


@pytest.mark.genesis
def test_oracle_lifts_cube(env):
    """The scripted expert grasps and lifts from the verified position."""
    from stable_worldmodel.envs.so101 import OraclePickLiftPolicy

    policy = OraclePickLiftPolicy()
    policy.set_env(env)
    obs, info = env.reset(
        seed=0,
        options={
            'variation_values': {'cube.start_position': np.array([0.15, 0.0])}
        },
    )
    infos = {'step_idx': np.array([0])}
    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        action = policy.get_action(infos)
        obs, reward, terminated, truncated, info = env.step(action)
        t += 1
        infos = {'step_idx': np.array([t])}
    assert terminated, 'oracle failed to lift the cube from (0.15, 0)'
    assert float(reward) > 0.05
