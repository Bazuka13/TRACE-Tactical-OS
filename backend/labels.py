# backend/labels.py run: python backend/labels.py (all matches in data/features)
import sys
import numpy as np
import pandas as pd
from config import FEATS, AUX, LABELS, TICK_RATE, DT

W2 = int(round(2.0 / DT))  # frames in 2 seconds (21)
W3 = int(round(3.0 / DT))  # frames in 3 seconds (32)
BUF = int(round(5.0 / DT))  # frames in 5 seconds (53)
NEG_PER_POS = 19  # about 5% positives
rng = np.random.default_rng(42)


def round_arrays(g):
    # one round -> arrays shaped (frames, 10 players, ...)
    g = g.sort_values(['tick', 'slot'])
    F = len(g) // 10

    def c(name, dtype=float):
        return g[name].to_numpy(dtype).reshape(F, 10)

    bits = c('los_bits', np.int64)
    return dict(
        ticks=g.tick.to_numpy()[::10],
        alive=c('is_alive', bool),
        hp=c('hp_pct'),
        pos=np.stack([c('pos_x'), c('pos_y')], axis=2),
        yaw=c('yaw'),
        team=c('team', int)[0],
        los=((bits[:, :, None] >> np.arange(10)) & 1).astype(bool)
    )  # los[f, i, j]: i sees j


def find_positives(r, hurt):
    # returns [(frame_index, victim_slot), ...] for the ambushes in one round
    out = []
    F = len(r['ticks'])
    if len(hurt) == 0:
        return out

    for (a, v), g in hurt.groupby(['att_slot', 'vic_slot']):
        if r['team'][a] == r['team'][v]:
            continue  # ignore friendly fire
        ev = np.sort(g['tick'].values)
        starts = ev[np.r_[True, np.diff(ev) > 3 * TICK_RATE]]  # first hit of each separate fight
        for st in starts:
            f0 = int(np.searchsorted(r['ticks'], st, side='right') - 1)  # last frame before the hit
            if f0 < W2 or f0 + 1 >= F or not r['alive'][f0, v]:
                continue
            vt = r['team'] == r['team'][v]
            if r['los'][f0 - W2:f0][:, vt, a].any():  # rule 1: someone on V's team saw A
                continue
            yaw = np.radians(r['yaw'][f0, v])
            face = np.array([np.cos(yaw), np.sin(yaw)])
            d = r['pos'][f0, a] - r['pos'][f0, v]
            d = d / (np.linalg.norm(d) + 1e-9)
            if np.degrees(np.arccos(np.clip(face @ d, -1, 1))) < 90:  # rule 2: A was in front
                continue
            seg = slice(f0, min(F, f0 + W3 + 1))
            hp_after = np.where(r['alive'][seg, v], r['hp'][seg, v], 0.0)
            if r['hp'][f0, v] - hp_after.min() < 0.5:  # rule 3: lost less than 50 HP
                continue
            out.append((f0, v))
    return out


def label_match(mid):
    pl_path = FEATS / (mid + '_players.parquet')
    hurt_path = AUX / (mid + '_hurt.parquet')
    if not pl_path.exists() or not hurt_path.exists():
        return None

    pl = pd.read_parquet(pl_path, columns=['round_num', 'tick', 'slot', 'team',
                                          'is_alive', 'hp_pct', 'pos_x', 'pos_y',
                                          'yaw', 'los_bits'])
    hurt = pd.read_parquet(hurt_path)
    pos_rows, neg_pool = [], []

    for rnd, g in pl.groupby('round_num'):
        r = round_arrays(g)
        if len(hurt) and 'tick' in hurt.columns:
            h = hurt[(hurt.tick >= r['ticks'][0]) & (hurt.tick <= r['ticks'][-1])]
        else:
            h = pd.DataFrame(columns=['tick', 'att_slot', 'vic_slot', 'dmg_health'])
        banned = np.zeros(len(r['ticks']), bool)
        for f0, v in find_positives(r, h):
            pos_rows.append((rnd, int(r['ticks'][f0]), int(v), 1))
            banned[max(0, f0 - BUF): f0 + BUF + 1] = True  # +-5 seconds are off limits
        fv = np.argwhere(r['alive'] & ~banned[:, None])  # every clean (frame, player)
        if len(fv):
            neg_pool.append(np.c_[np.full(len(fv), rnd), r['ticks'][fv[:, 0]], fv[:, 1]])

    if not pos_rows:
        return None

    neg = np.concatenate(neg_pool)
    pick = rng.choice(len(neg), min(len(neg), NEG_PER_POS * len(pos_rows)), replace=False)
    rows = pos_rows + [(int(a), int(b), int(c), 0) for a, b, c in neg[pick]]
    lab = pd.DataFrame(rows, columns=['round_num', 'tick', 'slot', 'y_ambush'])
    lab.insert(0, 'match_id', mid)
    return lab


if __name__ == '__main__':
    ids = sys.argv[1:] or sorted(p.name.replace('_players.parquet', '') for p in FEATS.glob('*_players.parquet'))
    for mid in ids:
        lab = label_match(mid)
        if lab is None:
            print('no ambushes found in', mid)
            continue
        lab.to_parquet(LABELS / (mid + '.parquet'))
        p = lab[lab.y_ambush == 1]
        print('%-10s positives=%d negatives=%d' % (mid, len(p), len(lab) - len(p)))
        for r in p.head(5).itertuples():
            print('   check in game: round %d, tick %d (console: demo_gototick %d)' % (r.round_num, r.tick, r.tick))
