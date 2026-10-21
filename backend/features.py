# backend/features.py
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from config import TICKS, AUX, FEATS, FEATURES, TICK_RATE, ROUND_SECONDS
from maps import units_to_norm, units_per_norm, load_walls

GRID = 64
_gx, _gy = np.meshgrid((np.arange(GRID) + 0.5) / GRID, (np.arange(GRID) + 0.5) / GRID)
CELLS = np.c_[_gx.ravel(), _gy.ravel()]  # centres of the 64 x 64 territory grid
IU, JU = np.triu_indices(10, 1)  # the 45 pairs of the 10 players


def blocked_by_walls(p1, p2, walls):
    # For every sight-line p1[k] -> p2[k]: True if it crosses at least one wall segment.
    if len(walls) == 0:
        return np.zeros(len(p1), dtype=bool)
    d1 = p2 - p1
    w3, w4 = walls[:, 0], walls[:, 1]
    d2 = (w4 - w3)[None]
    diff = w3[None] - p1[:, None]
    cross = d1[:, None, 0] * d2[..., 1] - d1[:, None, 1] * d2[..., 0]
    ok = np.abs(cross) > 1e-12  # parallel lines never cross
    safe = np.where(ok, cross, 1.0)
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / safe
    u = (diff[..., 0] * d1[:, None, 1] - diff[..., 1] * d1[:, None, 0]) / safe
    return (ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any(axis=1)


def los_matrix(pos, alive, walls, smokes):
    # 10 x 10 matrix: 1 = player i and player j can see each other, 0 = blocked or dead
    p1, p2 = pos[IU], pos[JU]
    blocked = blocked_by_walls(p1, p2, walls)
    d1 = p2 - p1
    for sx, sy, sr in smokes:  # a smoke blocks a ray that passes through its circle
        c = np.array([sx, sy])
        tt = np.clip(((c - p1) * d1).sum(1) / ((d1 ** 2).sum(1) + 1e-12), 0, 1)
        closest = p1 + tt[:, None] * d1
        blocked |= np.linalg.norm(closest - c, axis=1) < sr
    vis = (~blocked) & alive[IU] & alive[JU]
    los = np.zeros((10, 10), dtype=np.int8)
    los[IU, JU] = vis
    los[JU, IU] = vis
    return los


def frame_features(pos, yaw_r, alive, team, hp, wdps, walls, smokes, thr, lam):
    # All per-player features for ONE frame. Every array has 10 entries (one per slot).
    los = los_matrix(pos, alive, walls, smokes)
    seen = los.astype(bool)
    D = cdist(pos, pos)  # distances between all players (0..1 scale)
    both = alive[:, None] & alive[None, :]
    eye = np.eye(10, dtype=bool)
    enemy = (team[:, None] != team[None, :]) & both
    mate = (team[:, None] == team[None, :]) & both & ~eye

    f = {}
    f['n_enemies_visible'] = (seen & enemy).sum(1)
    f['n_teammates_visible'] = (seen & mate).sum(1)
    f['min_enemy_dist'] = np.where(enemy, D, 2.0).min(1)
    f['max_enemy_los_dist'] = np.where(seen & enemy, D, 0.0).max(1)

    # support graph: teammates who are close AND can see each other
    edge = mate & seen & (D <= thr)
    f['degree'] = edge.sum(1)
    f['is_isolated'] = (f['degree'] == 0).astype(int)
    face = np.c_[np.cos(np.radians(yaw_r)), np.sin(np.radians(yaw_r))]
    angle = np.degrees(np.arccos(np.clip(face @ face.T, -1, 1)))
    f['has_crossfire_partner'] = (edge & (angle >= 90)).any(1).astype(int)
    f['anchor_influence_score'] = (edge * (hp * wdps)[None, :]).sum(1)

    # hidden danger: living enemies that CANNOT see me now but are close (they could appear)
    f['nearest_danger_score'] = (np.exp(-lam * D) * (enemy & ~seen)).sum(1)

    # territory: every grid cell belongs to the nearest living player (Voronoi)
    idx = np.flatnonzero(alive)
    if len(idx):
        _, nn = cKDTree(pos[idx]).query(CELLS)
        pct_t = float((team[idx][nn] == 0).mean())
    else:
        pct_t = 0.5
    f['territory_control_pct'] = np.where(team == 0, pct_t, 1 - pct_t)

    # am I closer to an enemy than to any teammate? (nearest OTHER living player is an enemy)
    Dm = np.where(both & ~eye, D, np.inf)
    f['is_in_enemy_territory'] = (np.isfinite(Dm.min(1)) &
                                  (team[Dm.argmin(1)] != team)).astype(int)

    # remember who sees whom as a small number (bit j set = I can see player j)
    f['los_bits'] = (los.astype(np.int64) << np.arange(10)).sum(1)
    return f, pct_t


def side_stats(pos, alive, team, side):
    m = alive & (team == side)
    if not m.any():
        return [np.nan, np.nan, np.nan]
    c = pos[m].mean(axis=0)  # team centroid = average position
    return [c[0], c[1], np.linalg.norm(pos[m] - c, axis=1).std()]  # dispersion = how spread out


def process_match(match_id):
    df = pd.read_parquet(TICKS / (match_id + '.parquet'))
    smk_path = AUX / (match_id + '_smokes.parquet')
    if smk_path.exists():
        smk = pd.read_parquet(smk_path)
    else:
        smk = pd.DataFrame(columns=['tick_start', 'tick_end', 'x', 'y', 'r'])

    map_name = df.map_name.iloc[0]
    walls = load_walls(map_name)
    upn = units_per_norm(map_name)
    thr = units_to_norm(800, map_name)  # 'close enough to support' = 800 game units
    lam = 0.004 * upn  # distance decay, converted to the 0..1 scale

    df = df.sort_values(['round_num', 'tick', 'slot']).reset_index(drop=True)
    df = df[df.groupby(['round_num', 'tick']).slot.transform('size') == 10].reset_index(drop=True)
    F = len(df) // 10  # number of frames; each frame = 10 rows

    def col(name, dtype=float):
        return df[name].to_numpy(dtype).reshape(F, 10)

    pos = np.stack([col('pos_x'), col('pos_y')], axis=2)
    yaw, hp, wd, pw = col('yaw'), col('hp_pct'), col('weapon_dps'), col('team_power_idx')
    alive, team = col('is_alive', bool), col('team', int)
    ticks = df.tick.to_numpy()[::10]
    
    if len(smk) and 'tick_start' in smk.columns:
        smoke_arr = smk[['tick_start', 'tick_end', 'x', 'y', 'r']].to_numpy(dtype=float)
    else:
        smoke_arr = np.zeros((0, 5), dtype=float)

    rows, pct_t, st_att, st_def = [], np.zeros(F), [], []
    for f in range(F):
        if len(smoke_arr):
            s = smoke_arr[(smoke_arr[:, 0] <= ticks[f]) & (ticks[f] < smoke_arr[:, 1])]
            smokes = [(a[2], a[3], a[4]) for a in s]
        else:
            smokes = []
        feats, pct_t[f] = frame_features(pos[f], yaw[f], alive[f], team[f],
                                         hp[f], wd[f],
                                         walls, smokes, thr, lam)
        rows.append(feats)
        st_att.append(side_stats(pos[f], alive[f], team[f], 0))
        st_def.append(side_stats(pos[f], alive[f], team[f], 1))

    for k in rows[0]:
        df[k] = np.stack([r[k] for r in rows]).reshape(-1)

    df['los_bits'] = df['los_bits'].astype('int64')
    df['yaw_sin'] = np.sin(np.radians(df.yaw))
    df['yaw_cos'] = np.cos(np.radians(df.yaw))
    df['round_clock_norm'] = (df.round_clock_s / ROUND_SECONDS).clip(0, 1.5)
    df['round_phase_enc'] = np.digitize(df.round_clock_s, [30, 75])  # 0 early, 1 mid, 2 late

    # kinematics, computed inside each (round, player) so a new round never mixes with the last one
    g = df.groupby(['round_num', 'slot'])
    dt = g.tick.diff() / TICK_RATE  # real seconds between frames
    df['vel_x'] = (g.pos_x.diff() / dt * upn).fillna(0)
    df['vel_y'] = (g.pos_y.diff() / dt * upn).fillna(0)
    df['speed'] = np.hypot(df.vel_x, df.vel_y)
    g = df.groupby(['round_num', 'slot'])
    df['accel_mag'] = (np.hypot(g.vel_x.diff(), g.vel_y.diff()) / dt).fillna(0)
    df['speed_1s_mean'] = g.speed.transform(lambda s: s.rolling(10, min_periods=1).mean())
    df['speed_3s_mean'] = g.speed.transform(lambda s: s.rolling(30, min_periods=1).mean())

    keep = ['match_id', 'map_name', 'round_num', 'tick', 'slot', 'player_id',
            'team', 'round_winner', 'yaw', 'los_bits'] + FEATURES
    players = df[keep]

    # one row per frame for the whole teams (used by the win-probability model)
    fr = df.iloc[::10][['match_id', 'round_num', 'tick', 'round_clock_s',
                       'round_clock_norm', 'bomb_planted', 'round_winner']].reset_index(drop=True)
    fr['y_win'] = (fr.round_winner == 0).astype(int)  # 1 = attackers (T) won the round
    fr['att_power'] = np.nanmean(np.where(team == 0, pw, np.nan), axis=1)
    fr['def_power'] = np.nanmean(np.where(team == 1, pw, np.nan), axis=1)
    fr['power_delta'] = fr.att_power - fr.def_power
    fr['att_alive'] = (alive & (team == 0)).sum(1)
    fr['def_alive'] = (alive & (team == 1)).sum(1)
    fr['alive_diff'] = fr.att_alive - fr.def_alive
    fr['att_territory'] = pct_t
    a, d = np.array(st_att), np.array(st_def)
    fr['att_cx'], fr['att_cy'], fr['att_disp'] = a[:, 0], a[:, 1], a[:, 2]
    fr['def_disp'] = d[:, 2]
    g = fr.groupby('round_num')
    fr['centroid_speed'] = (np.hypot(g.att_cx.diff(), g.att_cy.diff())
                            / (g.tick.diff() / TICK_RATE) * upn).fillna(0)
    fr = fr.fillna(0)  # a wiped-out side has no centroid

    return players, fr


if __name__ == '__main__':
    ids = sys.argv[1:] or sorted(p.stem for p in TICKS.glob('*.parquet'))
    for mid in ids:
        players, team_df = process_match(mid)
        players.to_parquet(FEATS / (mid + '_players.parquet'))
        team_df.to_parquet(FEATS / (mid + '_team.parquet'))
        print('OK', mid, 'frames:', len(team_df), 'player rows:', len(players))
