"""SO-101 pick-cube demo: oracle rollouts, video, and data collection.

Mirrors scripts/examples/pusht.py for the Genesis-simulated SO-101 arm.
Requires the genesis extra: `uv sync --extra genesis`. The scripted
oracle succeeds on ~50-70% of in-basin cube placements (see
stable_worldmodel/envs/so101/expert_policy.py); failed grasps truncate
at the env's 200-step time limit, so filter demos by termination if you
need successes only.
"""

from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.envs.so101 import OraclePickLiftPolicy


VIDEO_DIR = Path(__file__).parent / 'videos' / 'so101'
DATA_DIR = Path(__file__).parent / 'data' / 'so101_demo.lance'

world = swm.World(
    'swm/SO101PickCube-v0',
    num_envs=2,
    image_shape=(224, 224),
    max_episode_steps=200,
    render_mode='rgb_array',
)
world.set_policy(OraclePickLiftPolicy())

# 1. Watch the oracle pick the cube up.
results = world.evaluate(episodes=2, seed=0, video=VIDEO_DIR)
print(f'evaluate results: {results}')

# 2. Collect a small dataset (lance format, append-friendly).
world.collect(DATA_DIR, episodes=10, seed=0)

# 3. Load it back and inspect.
ds = swm.data.load_dataset(str(DATA_DIR), num_steps=4)
print(f'dataset: {len(ds)} samples')
sample = ds[0]
for key, val in sample.items():
    shape = getattr(val, 'shape', None)
    print(f'  {key}: {shape if shape is not None else type(val).__name__}')
