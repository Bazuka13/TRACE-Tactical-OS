# backend/dataset.py
import numpy as np
import pandas as pd
from config import DATA, FEATS, LABELS, EMB


def match_ids(split):
    sp = pd.read_csv(DATA / 'splits.csv')
    return list(sp[sp.split == split].match_id)


def all_match_ids():
    return list(pd.read_csv(DATA / 'splits.csv').match_id)


def load_labeled(split, with_gnn=False):
    # one row per labelled (player, moment), with all 28 features (and optionally the 128 GNN numbers)
    parts = []
    for m in match_ids(split):
        if not (LABELS / (m + '.parquet')).exists():
            continue
        lab = pd.read_parquet(LABELS / (m + '.parquet'))
        pl = pd.read_parquet(FEATS / (m + '_players.parquet'))
        d = lab.merge(pl, on=['match_id', 'round_num', 'tick', 'slot'])
        if with_gnn:
            emb_file = EMB / (m + '_gnn.parquet')
            if not emb_file.exists():
                raise FileNotFoundError(f"Missing GNN embeddings: {emb_file}")
            d = d.merge(pd.read_parquet(emb_file), on=['match_id', 'round_num', 'tick'])
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"No labeled data found for split '{split}'")
    return pd.concat(parts, ignore_index=True)


def load_team(split):
    parts = []
    for m in match_ids(split):
        f = FEATS / (m + '_team.parquet')
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        raise FileNotFoundError(f"No team data found for split '{split}'")
    return pd.concat(parts, ignore_index=True)


def ece(y, p, bins=10):
    # Expected Calibration Error: when we say 70%, is it right about 70% of the time?
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total
