# backend/ingest.py
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from demoparser2 import DemoParser
except ImportError:
    DemoParser = None
from config import ROOT, DEMOS, TICKS, AUX, TICK_RATE, STEP
from maps import MAP_DATA, world_to_norm, units_to_norm

# rough firepower rating per weapon (0..1). Keys are lowercase with no spaces or dashes.
WEAPON_DPS = {
    'awp': 1.0, 'scar20': 0.9, 'g3sg1': 0.9, 'ak47': 0.85, 'm4a1': 0.8,
    'm4a4': 0.8,
    'sg553': 0.8, 'aug': 0.8, 'galil': 0.7, 'famas': 0.7, 'p90': 0.6,
    'negev': 0.6,
    'm249': 0.6, 'ssg08': 0.6, 'mp9': 0.55, 'mac10': 0.55, 'mp7': 0.55,
    'mp5': 0.55,
    'ump': 0.55, 'xm1014': 0.55, 'bizon': 0.5, 'nova': 0.5, 'mag7': 0.5,
    'sawedoff': 0.5,
    'deagle': 0.5, 'deserteagle': 0.5, 'revolver': 0.5, 'cz75': 0.35,
    'fiveseven': 0.3,
    'tec9': 0.3, 'p250': 0.3, 'elite': 0.3, 'dualberettas': 0.3, 'usp': 0.25,
    'glock': 0.25, 'p2000': 0.25,
}
unknown_weapons = set()


def weapon_rating(name):
    s = str(name).lower().replace('-', '').replace(' ', '').replace('_', '')
    for key, val in WEAPON_DPS.items():
        if key in s:
            return val
    unknown_weapons.add(s)  # knives, grenades, bomb, zeus ... get the default
    return 0.05


def winner_to_team(w):
    s = str(w).upper()
    if s in ('T', '2', 'TERRORIST'):
        return 0
    if s in ('CT', '3', 'COUNTERTERRORIST', 'COUNTER-TERRORIST'):
        return 1
    return np.nan


def build_rounds(parser):
    # one row per live round: when it starts (freeze time over), when it ends, who won, bomb plant time
    fe = parser.parse_event('round_freeze_end')[['tick']].rename(columns={'tick': 'start_tick'})
    en = parser.parse_event('round_end')[['tick', 'winner']].rename(columns={'tick': 'end_tick'})
    fe['start_tick'] = fe['start_tick'].astype('int64')
    en['end_tick'] = en['end_tick'].astype('int64')
    r = pd.merge_asof(fe.sort_values('start_tick'),
                      en.sort_values('end_tick'),
                      left_on='start_tick', right_on='end_tick',
                      direction='forward')
    r = r.dropna(subset=['end_tick']).drop_duplicates('end_tick', keep='last')
    r['winner_team'] = r['winner'].map(winner_to_team)
    r = r.dropna(subset=['winner_team']).reset_index(drop=True)
    r['round_num'] = np.arange(1, len(r) + 1)
    try:
        plants_ev = parser.parse_event('bomb_planted')
        plants = plants_ev['tick'].to_numpy() if len(plants_ev) and 'tick' in plants_ev.columns else np.array([])
    except Exception:
        plants = np.array([])
    r['plant_tick'] = [next((p for p in plants if s <= p <= e), np.nan)
                       for s, e in zip(r.start_tick, r.end_tick)]
    return r


def add_team_power(df):
    # Team Power Index = 0.35*cash + 0.30*alive + 0.20*armor + 0.15*weapon quality (all 0..1)
    df = df.copy()
    df['alive_i'] = df.is_alive.astype(float)
    df['armor_alive'] = df.armor_pct * df.alive_i
    df['wdps_alive'] = df.weapon_dps * df.alive_i
    g = df.groupby(['round_num', 'tick', 'team'])
    cash = g.cash.transform('sum')
    n = g.alive_i.transform('sum')
    den = n.clip(lower=1)
    df['team_power_idx'] = (
        0.35 * cash.clip(upper=25000) / 25000 +
        0.30 * n / 5 +
        0.20 * g.armor_alive.transform('sum') / den +
        0.15 * g.wdps_alive.transform('sum') / den
    )
    return df.drop(columns=['alive_i', 'armor_alive', 'wdps_alive'])


def parse_demo(demo_path):
    demo_path = Path(demo_path)
    match_id = demo_path.stem
    parser = DemoParser(str(demo_path))
    header = parser.parse_header()
    map_name = header['map_name']
    if map_name not in MAP_DATA:
        raise ValueError('map not supported yet: ' + map_name)
    rounds = build_rounds(parser)
    if len(rounds) == 0:
        raise ValueError('no complete rounds found in demo')

    # the frames we keep: every 6th tick from freeze-end to round-end, for every round
    parts = []
    for r in rounds.itertuples():
        t = np.arange(r.start_tick, r.end_tick + 1, STEP)
        parts.append(pd.DataFrame({
            'tick': t,
            'round_num': r.round_num,
            'round_clock_s': (t - r.start_tick) / TICK_RATE,
            'bomb_planted': (t >= r.plant_tick).astype(int) if not np.isnan(r.plant_tick) else 0,
            'round_winner': int(r.winner_team)
        }))
    frames = pd.concat(parts, ignore_index=True)

    props = ['X', 'Y', 'yaw', 'health', 'armor_value', 'team_num',
             'is_alive', 'balance', 'active_weapon_name']
    try:
        raw = parser.parse_ticks(props, ticks=frames.tick.tolist())
    except TypeError:
        # older versions: parse everything, then filter
        raw = parser.parse_ticks(props)
        raw = raw[raw.tick.isin(frames.tick)]
    except Exception as e:
        # Check if balance column has different name
        if 'balance' in str(e).lower():
            props = ['X', 'Y', 'yaw', 'health', 'armor_value', 'team_num',
                     'is_alive', 'active_weapon_name']
            raw = parser.parse_ticks(props, ticks=frames.tick.tolist())
            raw['balance'] = 4000.0
        else:
            raise

    if 'balance' not in raw.columns:
        raw['balance'] = 4000.0

    raw = raw[raw.team_num.isin([2, 3])].merge(frames, on='tick')
    
    # Filter to exactly the 10 primary match players if there are spectator/coach steamids
    id_counts = raw.steamid.value_counts()
    if len(id_counts) < 10:
        raise ValueError('expected 10 players, found %d' % len(id_counts))
    elif len(id_counts) > 10:
        ids = sorted(id_counts.head(10).index)
        raw = raw[raw.steamid.isin(ids)]
    else:
        ids = sorted(id_counts.index)

    slot_of = {sid: i for i, sid in enumerate(ids)}

    px, py = world_to_norm(raw.X, raw.Y, map_name)
    df = pd.DataFrame({
        'game': 'cs2',
        'match_id': match_id,
        'map_name': map_name,
        'round_num': raw.round_num.to_numpy(),
        'tick': raw.tick.to_numpy(),
        'round_clock_s': raw.round_clock_s.to_numpy(),
        'player_id': raw.steamid.astype('int64').to_numpy(),
        'slot': raw.steamid.map(slot_of).to_numpy(),
        'team': (raw.team_num == 3).astype(int).to_numpy(),  # 0 = T, 1 = CT
        'pos_x': px,
        'pos_y': py,
        'yaw': ((-raw.yaw) % 360).to_numpy(),  # radar-space angle
        'hp_pct': (raw.health / 100).clip(0, 1).to_numpy(),
        'armor_pct': (raw.armor_value / 100).clip(0, 1).to_numpy(),
        'weapon_dps': raw.active_weapon_name.map(weapon_rating).to_numpy(),
        'is_alive': raw.is_alive.astype(bool).to_numpy(),
        'cash': raw.balance.to_numpy(),
        'bomb_planted': raw.bomb_planted.to_numpy(),
        'round_winner': raw.round_winner.to_numpy()
    })
    n_before = len(df)
    df = df.dropna(subset=['pos_x', 'pos_y', 'yaw', 'hp_pct'])
    if len(df) < n_before:
        logging.warning('%s: dropped %d rows with NaN', match_id, n_before - len(df))
    df = add_team_power(df).sort_values(['round_num', 'tick', 'slot']).reset_index(drop=True)

    # Fix tick gaps: some demos have occasional server-dropped ticks.
    # Re-index each (round_num, slot) to the full tick grid and forward-fill.
    expected_ticks = frames[['round_num', 'tick']].drop_duplicates()
    slots = df.slot.unique()
    full_idx = expected_ticks.merge(pd.DataFrame({'slot': slots}), how='cross')
    df = full_idx.merge(df, on=['round_num', 'tick', 'slot'], how='left')
    df = df.sort_values(['round_num', 'tick', 'slot']).reset_index(drop=True)
    # Forward-fill per (round_num, slot): carry last known state for dropped ticks
    fill_cols = [c for c in df.columns if c not in ('round_num', 'tick', 'slot')]
    df[fill_cols] = df.groupby(['round_num', 'slot'])[fill_cols].ffill()
    # Backward-fill any NaN at the start of a round (rare: first tick dropped)
    df[fill_cols] = df.groupby(['round_num', 'slot'])[fill_cols].bfill()
    df = df.dropna(subset=['pos_x', 'pos_y'])
    # Re-derive round_clock_s from the tick grid (it may have been NaN for filled rows)
    round_starts = frames.groupby('round_num').tick.min().to_dict()
    df['round_clock_s'] = df.apply(lambda r: (r.tick - round_starts[r.round_num]) / TICK_RATE, axis=1)

    # smoke grenades: each lasts about 18 seconds, radius about 144 game units
    try:
        sm = parser.parse_event('smokegrenade_detonate')
        if len(sm) and 'x' in sm.columns:
            sx, sy = world_to_norm(sm.x, sm.y, map_name)
            smokes = pd.DataFrame({
                'tick_start': sm.tick.to_numpy(),
                'tick_end': sm.tick.to_numpy() + 18 * TICK_RATE,
                'x': sx,
                'y': sy,
                'r': units_to_norm(144, map_name)
            })
        else:
            smokes = pd.DataFrame(columns=['tick_start', 'tick_end', 'x', 'y', 'r'])
    except Exception:
        smokes = pd.DataFrame(columns=['tick_start', 'tick_end', 'x', 'y', 'r'])

    # who hurt whom (used in Phase 2 to find ambushes)
    try:
        h_ev = parser.parse_event('player_hurt')
        if len(h_ev) and 'tick' in h_ev.columns:
            h = h_ev[['tick', 'attacker_steamid', 'user_steamid', 'dmg_health']].dropna()
            h['att_slot'] = pd.to_numeric(h.attacker_steamid, errors='coerce').astype('Int64').map(slot_of)
            h['vic_slot'] = pd.to_numeric(h.user_steamid, errors='coerce').astype('Int64').map(slot_of)
            hurt = h.dropna(subset=['att_slot', 'vic_slot']).astype({'att_slot': int, 'vic_slot': int})
            hurt = hurt[['tick', 'att_slot', 'vic_slot', 'dmg_health']]
        else:
            hurt = pd.DataFrame(columns=['tick', 'att_slot', 'vic_slot', 'dmg_health'])
    except Exception:
        hurt = pd.DataFrame(columns=['tick', 'att_slot', 'vic_slot', 'dmg_health'])

    return df, smokes, hurt, rounds


def validate(df):
    # the Phase 0 exit checks
    assert df[['pos_x', 'pos_y', 'yaw', 'hp_pct']].notna().all().all(), 'NaN found'
    inside = df.pos_x.between(-0.02, 1.02) & df.pos_y.between(-0.02, 1.02)
    assert inside.mean() > 0.99, 'positions fall outside the map. Wrong MAP_DATA? inside=%.3f' % inside.mean()
    assert df.hp_pct.between(0, 1).all() and df.armor_pct.between(0, 1).all()
    assert df.team_power_idx.between(0, 1).all(), 'team_power_idx outside 0..1'
    step = df.sort_values('tick').groupby(['round_num', 'slot']).tick.diff().dropna()
    assert (step == STEP).all(), 'ticks are not evenly spaced'
    assert df.groupby(['round_num', 'tick']).size().eq(10).mean() > 0.99, 'frames missing players'


def save(demo_path):
    df, smokes, hurt, rounds = parse_demo(demo_path)
    validate(df)
    mid = Path(demo_path).stem
    df.to_parquet(TICKS / (mid + '.parquet'))
    smokes.to_parquet(AUX / (mid + '_smokes.parquet'))
    hurt.to_parquet(AUX / (mid + '_hurt.parquet'))
    rounds.to_parquet(AUX / (mid + '_rounds.parquet'))
    print('OK %-12s map=%s rounds=%d rows=%d' % (
        mid, df.map_name.iloc[0], df.round_num.nunique(), len(df)))


if __name__ == '__main__':
    logging.basicConfig(filename=str(ROOT / 'ingest_errors.log'), level=logging.INFO)
    paths = [Path(a) for a in sys.argv[1:]] or sorted(DEMOS.glob('*.dem'))
    for p in paths:
        try:
            save(p)
        except Exception as e:
            # one bad demo must not stop the batch
            print('FAILED', p.name, '->', e)
            logging.exception(p.name)
    if unknown_weapons:
        print('Weapons rated as default 0.05 (check none are real guns):', sorted(unknown_weapons))
