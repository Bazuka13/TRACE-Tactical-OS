import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np
import pandas as pd
from config import LABELS
from labels import find_positives


def fake_round():
    F = 200
    r = dict(ticks=np.arange(F) * 6, alive=np.ones((F, 10), bool),
             hp=np.ones((F, 10)),
             pos=np.zeros((F, 10, 2)), yaw=np.zeros((F, 10)),
             team=np.array([0] * 5 + [1] * 5), los=np.zeros((F, 10, 10), bool))
    r['pos'][:, 7] = [0.8, 0.5]  # attacker, slot 7
    r['pos'][:, 1] = [0.2, 0.5]  # victim, slot 1
    r['yaw'][:, 1] = 180  # victim looks away from the attacker
    r['hp'][110:, 1] = 0.2  # victim loses 80 HP after the hit
    hurt = pd.DataFrame(dict(att_slot=[7, 7], vic_slot=[1, 1], tick=[602, 610]))
    return r, hurt


def test_real_ambush_is_found():
    r, hurt = fake_round()
    assert find_positives(r, hurt) == [(100, 1)]


def test_seen_attacker_is_not_an_ambush():
    r, hurt = fake_round()
    r['los'][90:100, 0, 7] = True  # a teammate saw the attacker just before
    assert find_positives(r, hurt) == []


def test_no_negative_near_a_positive():
    files = list(LABELS.glob('*.parquet'))
    if len(files) == 0:
        return
    for f in files:
        d = pd.read_parquet(f)
        pos, neg = d[d.y_ambush == 1], d[d.y_ambush == 0]
        m = neg.merge(pos[['round_num', 'tick']], on='round_num', suffixes=('', '_p'))
        assert not ((m.tick - m.tick_p).abs() <= 5 * 64).any(), f.name
