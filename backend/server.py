# backend/server.py run: uvicorn server:app --app-dir backend --port 8000
import asyncio
import os
import re
import numpy as np
import orjson
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import cKDTree
from config import FEATS, AUX, EMB, DT
from maps import load_walls, units_per_norm
from features import CELLS, GRID, blocked_by_walls
from explain import load_models, Explainers, threat_sentence, win_sentence

ALLOWED = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]
MAX_CLIENTS = int(os.environ.get('MAX_CLIENTS', '5'))
SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
active = {'n': 0}

app = FastAPI(title="Esports Tactical OS - Pro Analytics Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED if '*' not in ALLOWED else ['*'],
    allow_methods=['GET'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'ok': True, 'clients_active': active['n'], 'max_clients': MAX_CLIENTS}


# Models loaded lazily
M = None
EXPL = None
USES_GNN = False

DG = 32  # the danger overlay uses a coarser 32 x 32 grid for speed
_dx, _dy = np.meshgrid((np.arange(DG) + 0.5) / DG, (np.arange(DG) + 0.5) / DG)
DCELLS = np.c_[_dx.ravel(), _dy.ravel()]

# Map objective landmarks on de_mirage (normalized coordinates)
SITE_A_POS = np.array([0.68, 0.38])
SITE_B_POS = np.array([0.38, 0.72])


def get_models():
    global M, EXPL, USES_GNN
    if M is None:
        M = load_models()
        EXPL = Explainers(M)
        USES_GNN = any(c[0] == 'g' and c[1:].isdigit() for c in M['a_cols'])
    return M, EXPL, USES_GNN


@app.get('/matches')
def get_matches():
    files = sorted(p.name.replace('_players.parquet', '') for p in FEATS.glob('*_players.parquet'))
    return files


@app.get('/matches/{match_id}/rounds')
def get_match_rounds(match_id: str):
    pl_path = FEATS / (match_id + '_players.parquet')
    if not pl_path.exists():
        return {'rounds': []}
    pl = pd.read_parquet(pl_path, columns=['round_num'])
    return {'rounds': sorted(int(r) for r in pl.round_num.unique())}


@app.get('/matches/{match_id}/{round_num}/summary')
def get_round_summary(match_id: str, round_num: int):
    pl_path = FEATS / (match_id + '_players.parquet')
    if not pl_path.exists():
        return {'error': 'Match not found'}
    pl = pd.read_parquet(pl_path)
    pl = pl[pl.round_num == round_num]
    if len(pl) == 0:
        return {'error': 'Round not found'}
    
    rnd_winner = int(pl.round_winner.iloc[0])
    ticks = pl.tick.unique()
    return {
        'match_id': match_id,
        'round_num': round_num,
        'total_ticks': len(ticks),
        'start_tick': int(ticks.min()),
        'end_tick': int(ticks.max()),
        'winner_team': rnd_winner,
        'winner_name': 'Terrorists (Attack)' if rnd_winner == 0 else 'Counter-Terrorists (Defense)'
    }


def territory_grid(pos, alive, team):
    idx = np.flatnonzero(alive)
    if len(idx) == 0:
        return np.full((GRID, GRID), -1, dtype=np.int8)
    _, nn = cKDTree(pos[idx]).query(CELLS)
    return team[idx][nn].reshape(GRID, GRID).astype(np.int8)


def danger_grid(pos, alive, threat, walls, lam):
    grid = np.zeros(len(DCELLS))
    for j in np.flatnonzero(alive & (threat >= 0.2)):
        p = np.repeat(pos[j][None], len(DCELLS), axis=0)
        visible = ~blocked_by_walls(p, DCELLS, walls)
        d = np.linalg.norm(DCELLS - pos[j], axis=1)
        grid += threat[j] * visible * np.exp(-lam * d)
    return np.clip(grid, 0, 1).reshape(DG, DG)


def load_round(match_id, round_num):
    _, _, uses_gnn = get_models()
    pl = pd.read_parquet(FEATS / (match_id + '_players.parquet'))
    pl = pl[pl.round_num == round_num]
    tm = pd.read_parquet(FEATS / (match_id + '_team.parquet'))
    tm = tm[tm.round_num == round_num].set_index('tick')
    if uses_gnn:
        pl = pl.merge(pd.read_parquet(EMB / (match_id + '_gnn.parquet')),
                      on=['match_id', 'round_num', 'tick'])
    return pl.sort_values(['tick', 'slot']), tm


def load_round_events(match_id, min_tick, max_tick, pl):
    # Smokes
    smokes_list = []
    smk_path = AUX / f'{match_id}_smokes.parquet'
    if smk_path.exists():
        smk = pd.read_parquet(smk_path)
        r_smk = smk[(smk.tick_end >= min_tick) & (smk.tick_start <= max_tick)]
        for s in r_smk.itertuples():
            smokes_list.append({
                'tick_start': int(s.tick_start),
                'tick_end': int(s.tick_end),
                'x': round(float(s.x), 4),
                'y': round(float(s.y), 4),
                'r': round(float(s.r), 4)
            })

    # Damage and Kills
    hurt_path = AUX / f'{match_id}_hurt.parquet'
    h_df = pd.DataFrame()
    if hurt_path.exists():
        hurt = pd.read_parquet(hurt_path)
        h_df = hurt[(hurt.tick >= min_tick) & (hurt.tick <= max_tick)]

    # Compute exact death events from player alive state transitions
    deaths = []
    for slot, g in pl.groupby('slot'):
        alive_seq = g.is_alive.values
        ticks = g.tick.values
        xs, ys = g.pos_x.values, g.pos_y.values
        teams = g.team.values
        for i in range(1, len(alive_seq)):
            if alive_seq[i - 1] and not alive_seq[i]:
                d_tick = int(ticks[i])
                killer_slot = -1
                if len(h_df):
                    recent = h_df[(h_df.vic_slot == slot) & (abs(h_df.tick - d_tick) <= 128)]
                    if len(recent):
                        killer_slot = int(recent.iloc[-1].att_slot)
                deaths.append({
                    'slot': int(slot),
                    'team': int(teams[i]),
                    'tick': d_tick,
                    'x': round(float(xs[i]), 4),
                    'y': round(float(ys[i]), 4),
                    'killer': killer_slot
                })

    return smokes_list, deaths


def predict_frame(g, trow):
    models, _, _ = get_models()
    alive = g.is_alive.to_numpy(bool)
    threat = np.zeros(len(g))
    if alive.any():
        raw = models['cb'].predict_proba(g.loc[alive, models['a_cols']])[:, 1]
        threat[alive] = models['cb_cal'].transform(raw)
    xw = trow[models['w_cols']].to_frame().T.astype(float)
    wp = float(np.clip(models['w_cal'].transform(models['lg'].predict(xw))[0], 0, 1))
    return alive, threat, wp, xw


def compute_xai(Xp, slots, probs, xw, wp):
    _, expl, _ = get_models()
    players = {}
    if len(Xp):
        for slot, p, r in zip(slots, probs, expl.players(Xp)):
            players[int(slot)] = {
                'text': threat_sentence(int(slot), float(p), r),
                'factors': r
            }
    win_reasons = expl.win(xw)
    return {
        'players': players,
        'win': {
            'text': win_sentence(wp, win_reasons),
            'factors': win_reasons
        }
    }


def compute_extra_analytics(trow, threat, alive, team, bomb_planted):
    att_power = float(trow.get('att_power', 0.5))
    def_power = float(trow.get('def_power', 0.5))
    att_alive = int(trow.get('att_alive', 5))
    def_alive = int(trow.get('def_alive', 5))
    att_cx = float(trow.get('att_cx', 0.5))
    att_cy = float(trow.get('att_cy', 0.5))

    # 1. Economy Tiers
    tier_t = 'FULL BUY' if att_power >= 0.65 else 'FORCE BUY' if att_power >= 0.35 else 'ECO'
    tier_ct = 'FULL BUY' if def_power >= 0.65 else 'FORCE BUY' if def_power >= 0.35 else 'ECO'

    # 2. Team Ambush Risk
    t_threats = threat[alive & (team == 0)]
    ct_threats = threat[alive & (team == 1)]
    risk_t = round(float(t_threats.max()), 3) if len(t_threats) else 0.0
    risk_ct = round(float(ct_threats.max()), 3) if len(ct_threats) else 0.0

    # 3. Site Attack Prediction (Mirage Site A vs Site B)
    dist_a = float(np.hypot(att_cx - SITE_A_POS[0], att_cy - SITE_A_POS[1]))
    dist_b = float(np.hypot(att_cx - SITE_B_POS[0], att_cy - SITE_B_POS[1]))
    site_prob_a = round(dist_b / (dist_a + dist_b + 1e-6), 2)
    predicted_site = 'SITE A' if site_prob_a >= 0.5 else 'SITE B'

    # 4. Retake Odds (Post-plant dynamics)
    if bomb_planted == 1:
        # Post plant: defense needs to retake
        retake_odds = round(min(0.95, max(0.05, (def_alive / max(1, att_alive + def_alive)) * 0.7)), 2)
    else:
        # Pre plant: chance of successful plant
        retake_odds = round(min(0.90, max(0.10, att_alive / max(1, att_alive + def_alive))), 2)

    # 5. Clutch Odds
    clutch_info = None
    if att_alive == 1 and def_alive >= 2:
        odds = round(0.5 ** def_alive, 2)
        clutch_info = {'clutcher': 'T', 'situation': f'1v{def_alive}', 'odds': odds}
    elif def_alive == 1 and att_alive >= 2:
        odds = round(0.5 ** att_alive, 2)
        clutch_info = {'clutcher': 'CT', 'situation': f'1v{att_alive}', 'odds': odds}

    return {
        'economy_tier_t': tier_t,
        'economy_tier_ct': tier_ct,
        'risk_t': risk_t,
        'risk_ct': risk_ct,
        'predicted_site': predicted_site,
        'site_prob_a': site_prob_a,
        'retake_odds': retake_odds,
        'clutch': clutch_info
    }


def build_frame_dict(i, total_frames, tick, g, trow, alive, threat, wp, cache, map_name, walls, lam, round_smokes, round_deaths):
    pos = g[['pos_x', 'pos_y']].to_numpy()
    team = g.team.to_numpy()
    bomb_planted = int(trow['bomb_planted'])

    # Active smokes for this exact frame
    active_smokes = []
    for s in round_smokes:
        if s['tick_start'] <= tick <= s['tick_end']:
            dur = max(1, s['tick_end'] - s['tick_start'])
            prog = (tick - s['tick_start']) / dur
            active_smokes.append({
                'x': s['x'],
                'y': s['y'],
                'r': s['r'],
                'progress': round(prog, 2)
            })

    # Cumulative deaths up to this frame + recent kills within last ~2.5s (28 ticks)
    all_deaths_now = []
    recent_kills = []
    for d in round_deaths:
        if d['tick'] <= tick:
            all_deaths_now.append(d)
        if abs(tick - d['tick']) <= 28 and d['tick'] <= tick:
            recent_kills.append(d)

    # Extra predictive telemetry
    extra = compute_extra_analytics(trow, threat, alive, team, bomb_planted)

    upn = units_per_norm(map_name)
    players = []
    for k, p in enumerate(g.itertuples()):
        dist_enemy = -1.0
        if bool(p.is_alive):
            enemy_pos = [pos[j] for j, al in enumerate(alive) if al and team[j] != p.team]
            if enemy_pos:
                dist_norm = float(np.min(np.linalg.norm(np.array(enemy_pos) - np.array([p.pos_x, p.pos_y]), axis=1)))
                dist_enemy = round(dist_norm * upn, 1)

        players.append({
            'slot': int(p.slot),
            'team': int(p.team),
            'x': round(float(p.pos_x), 4),
            'y': round(float(p.pos_y), 4),
            'yaw': round(float(p.yaw), 1),
            'hp': round(float(p.hp_pct), 2),
            'armor': round(float(getattr(p, 'armor_pct', 0.0)), 2),
            'weapon_dps': round(float(getattr(p, 'weapon_dps', 0.0)), 2),
            'speed': round(float(getattr(p, 'speed', 0.0)), 1),
            'alive': bool(p.is_alive),
            'threat': round(float(threat[k]), 3),
            'dist_enemy': dist_enemy,
            'los_bits': int(getattr(p, 'los_bits', 0)),
            'is_isolated': int(getattr(p, 'is_isolated', 0)),
            'has_crossfire': int(getattr(p, 'has_crossfire_partner', 0)),
            'in_enemy_zone': int(getattr(p, 'is_in_enemy_territory', 0)),
            'why': cache['players'].get(int(p.slot), {}).get('text', ''),
            'factors': cache['players'].get(int(p.slot), {}).get('factors', [])
        })

    return {
        'type': 'frame',
        'frame_idx': i,
        'total_frames': total_frames,
        'tick': int(tick),
        'map': map_name,
        'clock': float(trow['round_clock_s']),
        'players': players,
        'territory': territory_grid(pos, alive, team).tolist(),
        'danger': np.round(danger_grid(pos, alive, threat, walls, lam), 2).tolist(),
        'win_prob': round(wp, 4),
        'why_win': cache['win'].get('text', ''),
        'win_factors': cache['win'].get('factors', []),
        'bomb_planted': bomb_planted,
        'att_power': round(float(trow.get('att_power', 0.0)), 2),
        'def_power': round(float(trow.get('def_power', 0.0)), 2),
        'att_alive': int(trow.get('att_alive', (alive & (team == 0)).sum())),
        'def_alive': int(trow.get('def_alive', (alive & (team == 1)).sum())),
        'att_territory': round(float(trow.get('att_territory', 0.5)), 3),
        'smokes': active_smokes,
        'deaths': all_deaths_now,
        'recent_kills': recent_kills,
        'extra': extra
    }


@app.get('/matches/{match_id}/{round_num}/initial_frame')
def get_initial_frame(match_id: str, round_num: int):
    try:
        models, _, _ = get_models()
        pl, tm = load_round(match_id, round_num)
        map_name = pl.map_name.iloc[0]
    except Exception as e:
        return {'error': str(e)}

    walls = load_walls(map_name)
    lam = 0.004 * units_per_norm(map_name)
    grouped_frames = list(pl.groupby('tick'))
    total_frames = len(grouped_frames)
    min_tick = int(pl.tick.min())
    max_tick = int(pl.tick.max())
    round_smokes, round_deaths = load_round_events(match_id, min_tick, max_tick, pl)

    tick0, g0 = grouped_frames[0]
    trow0 = tm.loc[tick0]
    alive, threat, wp, xw = predict_frame(g0, trow0)

    # Compute initial XAI explanations for frame 0
    init_xai = compute_xai(g0.loc[alive, models['a_cols']],
                           g0.slot.to_numpy()[alive],
                           threat[alive], xw, wp)

    payload = build_frame_dict(0, total_frames, tick0, g0, trow0, alive, threat, wp,
                               init_xai, map_name, walls, lam, round_smokes, round_deaths)

    return {
        'meta': {
            'type': 'meta',
            'match_id': match_id,
            'round_num': round_num,
            'map': map_name,
            'total_frames': total_frames,
            'total_deaths': len(round_deaths),
            'walls': [[[float(pt[0]), float(pt[1])] for pt in w] for w in walls]
        },
        'frame': payload
    }


@app.websocket('/ws/replay/{match_id}/{round_num}')
async def replay(ws: WebSocket, match_id: str, round_num: int):
    origin = ws.headers.get('origin')
    if origin is not None and '*' not in ALLOWED and origin not in ALLOWED:
        await ws.close(code=1008)
        return
    if not SAFE_ID.match(match_id) or active['n'] >= MAX_CLIENTS:
        await ws.close(code=1008)
        return

    await ws.accept()
    active['n'] += 1
    try:
        models, _, _ = get_models()
        pl, tm = load_round(match_id, round_num)
        map_name = pl.map_name.iloc[0]
    except Exception as e:
        await ws.send_text(orjson.dumps({'error': str(e)}).decode())
        await ws.close()
        active['n'] -= 1
        return

    walls = load_walls(map_name)
    lam = 0.004 * units_per_norm(map_name)
    cache = {'players': {}, 'win': {'text': '', 'factors': []}}
    busy = {'v': False}
    tasks = set()
    loop = asyncio.get_running_loop()

    grouped_frames = list(pl.groupby('tick'))
    total_frames = len(grouped_frames)
    min_tick = int(pl.tick.min())
    max_tick = int(pl.tick.max())

    # Pre-extract round-wide smokes and deaths
    round_smokes, round_deaths = load_round_events(match_id, min_tick, max_tick, pl)

    # Initial metadata payload
    await ws.send_text(orjson.dumps({
        'type': 'meta',
        'match_id': match_id,
        'round_num': round_num,
        'map': map_name,
        'total_frames': total_frames,
        'total_deaths': len(round_deaths),
        'walls': [[[float(pt[0]), float(pt[1])] for pt in w] for w in walls]
    }).decode())

    async def refresh(*args):
        busy['v'] = True
        try:
            cache.update(await loop.run_in_executor(None, compute_xai, *args))
        finally:
            busy['v'] = False

    speed_factor = 1.0
    seek_target = None
    stream_active = True

    async def client_listener():
        nonlocal speed_factor, seek_target, stream_active
        try:
            while stream_active:
                text = await ws.receive_text()
                msg = orjson.loads(text)
                mtype = msg.get('type')
                if mtype == 'set_speed':
                    speed_factor = max(0.2, min(5.0, float(msg.get('speed', 1.0))))
                elif mtype == 'seek':
                    seek_target = max(0, min(total_frames - 1, int(msg.get('frame', 0))))
        except Exception:
            pass

    listener_task = asyncio.create_task(client_listener())

    try:
        curr_idx = 0
        while curr_idx < total_frames:
            if seek_target is not None:
                curr_idx = seek_target
                seek_target = None

            tick, g = grouped_frames[curr_idx]
            started = loop.time()
            trow = tm.loc[tick]
            alive, threat, wp, xw = predict_frame(g, trow)

            # Refresh SHAP intelligence every 5 frames in background thread
            if curr_idx % 5 == 0 and not busy['v']:
                t = asyncio.create_task(refresh(g.loc[alive, models['a_cols']],
                                                g.slot.to_numpy()[alive],
                                                threat[alive], xw, wp))
                tasks.add(t)
                t.add_done_callback(tasks.discard)

            payload = build_frame_dict(curr_idx, total_frames, tick, g, trow, alive, threat, wp,
                                       cache, map_name, walls, lam, round_smokes, round_deaths)
            await ws.send_text(orjson.dumps(payload).decode())

            target_delay = DT / max(0.2, speed_factor)
            await asyncio.sleep(max(0.0, target_delay - (loop.time() - started)))
            curr_idx += 1

        await ws.send_text(orjson.dumps({'done': True}).decode())
    except WebSocketDisconnect:
        pass
    finally:
        active['n'] = max(0, active['n'] - 1)
        stream_active = False
        listener_task.cancel()
