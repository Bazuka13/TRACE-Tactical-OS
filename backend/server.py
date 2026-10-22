# backend/server.py run: uvicorn server:app --app-dir backend --port 8000
import asyncio
import numpy as np
import orjson
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import cKDTree
from config import FEATS, EMB, DT
from maps import load_walls, units_per_norm
from features import CELLS, GRID, blocked_by_walls
from explain import load_models, Explainers, threat_sentence, win_sentence

app = FastAPI(title="Esports Tactical OS Replay Server")
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])

# Models loaded lazily
M = None
EXPL = None
USES_GNN = False

DG = 32  # the danger overlay uses a coarser 32 x 32 grid for speed
_dx, _dy = np.meshgrid((np.arange(DG) + 0.5) / DG, (np.arange(DG) + 0.5) / DG)
DCELLS = np.c_[_dx.ravel(), _dy.ravel()]


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
            players[int(slot)] = {'text': threat_sentence(int(slot), float(p), r)}
    return {'players': players, 'win': {'text': win_sentence(wp, expl.win(xw))}}


@app.websocket('/ws/replay/{match_id}/{round_num}')
async def replay(ws: WebSocket, match_id: str, round_num: int):
    await ws.accept()
    try:
        models, _, _ = get_models()
        pl, tm = load_round(match_id, round_num)
        map_name = pl.map_name.iloc[0]
    except Exception as e:
        await ws.send_text(orjson.dumps({'error': str(e)}).decode())
        await ws.close()
        return

    walls = load_walls(map_name)
    lam = 0.004 * units_per_norm(map_name)
    cache = {'players': {}, 'win': {'text': ''}}
    busy = {'v': False}
    tasks = set()
    loop = asyncio.get_running_loop()

    grouped_frames = list(pl.groupby('tick'))
    total_frames = len(grouped_frames)

    # Initial handshake with map geometry and frame count
    await ws.send_text(orjson.dumps({
        'type': 'meta',
        'match_id': match_id,
        'round_num': round_num,
        'map': map_name,
        'total_frames': total_frames,
        'walls': [[[float(pt[0]), float(pt[1])] for pt in w] for w in walls]
    }).decode())

    async def refresh(*args):
        busy['v'] = True
        try:
            cache.update(await loop.run_in_executor(None, compute_xai, *args))
        finally:
            busy['v'] = False

    try:
        for i, (tick, g) in enumerate(grouped_frames):
            started = loop.time()
            trow = tm.loc[tick]
            alive, threat, wp, xw = predict_frame(g, trow)
            if i % 5 == 0 and not busy['v']:
                t = asyncio.create_task(refresh(g.loc[alive, models['a_cols']],
                                                g.slot.to_numpy()[alive],
                                                threat[alive], xw, wp))
                tasks.add(t)
                t.add_done_callback(tasks.discard)

            pos = g[['pos_x', 'pos_y']].to_numpy()
            team = g.team.to_numpy()
            players = [
                {
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
                    'los_bits': int(getattr(p, 'los_bits', 0)),
                    'is_isolated': int(getattr(p, 'is_isolated', 0)),
                    'has_crossfire': int(getattr(p, 'has_crossfire_partner', 0)),
                    'why': cache['players'].get(int(p.slot), {}).get('text', '')
                }
                for k, p in enumerate(g.itertuples())
            ]

            payload = {
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
                'why_win': cache['win']['text'],
                'bomb_planted': int(trow['bomb_planted']),
                'att_power': round(float(trow.get('att_power', 0.0)), 2),
                'def_power': round(float(trow.get('def_power', 0.0)), 2),
                'att_alive': int(trow.get('att_alive', (alive & (team == 0)).sum())),
                'def_alive': int(trow.get('def_alive', (alive & (team == 1)).sum())),
                'att_territory': round(float(trow.get('att_territory', 0.5)), 3)
            }
            await ws.send_text(orjson.dumps(payload).decode())
            await asyncio.sleep(max(0.0, DT - (loop.time() - started)))
        await ws.send_text(orjson.dumps({'done': True}).decode())
    except WebSocketDisconnect:
        pass
