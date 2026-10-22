import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np
import pandas as pd
from config import FEATS, FEATURES
from features import los_matrix


def test_wall_blocks_sight():
    pos = np.zeros((10, 2))
    pos[0], pos[1], pos[2] = [0.2, 0.5], [0.8, 0.5], [0.2, 0.9]
    alive = np.zeros(10, bool)
    alive[:3] = True
    walls = np.array([[[0.5, 0.0], [0.5, 1.0]]])  # a wall between player 0 and 1
    los = los_matrix(pos, alive, walls, [])
    assert los[0, 1] == 0 and los[1, 0] == 0  # blocked, and symmetric
    assert los[0, 2] == 1  # same side of the wall: clear


def test_smoke_blocks_sight():
    pos = np.zeros((10, 2))
    pos[0], pos[1] = [0.2, 0.5], [0.8, 0.5]
    alive = np.zeros(10, bool)
    alive[:2] = True
    no_walls = np.zeros((0, 2, 2))
    assert los_matrix(pos, alive, no_walls, [(0.5, 0.5, 0.05)])[0, 1] == 0
    assert los_matrix(pos, alive, no_walls, [])[0, 1] == 1


def test_feature_files_are_complete():
    files = list(FEATS.glob('*_players.parquet'))
    if len(files) == 0:
        return
    for f in files:
        df = pd.read_parquet(f)
        assert not df[FEATURES].isna().any().any()
        assert set(FEATURES) <= set(df.columns)
