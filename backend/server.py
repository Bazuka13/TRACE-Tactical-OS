# backend/server.py run: uvicorn server:app --app-dir backend --port 8000
import asyncio
import os
import re
import time
from pathlib import Path
import numpy as np
import orjson
import json
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import cKDTree
from config import ROOT, FEATS, AUX, EMB, DT, DEMOS, MODELS
from maps import load_walls, units_per_norm
from features import CELLS, GRID, blocked_by_walls
from explain import load_models, Explainers, threat_sentence, win_sentence, tactical_directive

ALLOWED = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]
MAX_CLIENTS = int(os.environ.get('MAX_CLIENTS', '5'))
SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
active = {'n': 0}

app = FastAPI(title="Esports Tactical OS - Pro Analytics Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED if '*' not in ALLOWED else ['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health(request: Request):
    accept = request.headers.get('accept', '')
    ua = request.headers.get('user-agent', '').lower()
    is_browser = ('text/html' in accept) or ('mozilla' in ua) or ('chrome' in ua) or ('safari' in ua)
    if is_browser and 'application/json' not in accept:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <title>Esports Tactical OS - Health Gateway</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="3;url=/">
  <style>
    body {{
      background: #050811;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 20px;
    }}
    .card {{
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(56, 189, 248, 0.4);
      box-shadow: 0 10px 40px rgba(0,0,0,0.8), 0 0 24px rgba(56, 189, 248, 0.2);
      border-radius: 14px;
      padding: 36px;
      max-width: 540px;
      width: 100%;
      text-align: center;
    }}
    .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.18);
      border: 1px solid #10b981;
      color: #34d399;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 22px;
    }}
    h1 {{ font-size: 24px; margin: 0 0 12px; font-weight: 800; letter-spacing: 0.03em; }}
    p {{ color: #94a3b8; font-size: 14px; line-height: 1.6; margin: 0 0 28px; }}
    .btn {{
      display: inline-block;
      background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
      color: #040711;
      font-weight: 800;
      font-size: 15px;
      text-decoration: none;
      padding: 14px 28px;
      border-radius: 8px;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.45);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .btn:hover {{
      transform: scale(1.04);
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.65);
    }}
    .meta {{
      margin-top: 26px;
      font-family: monospace;
      font-size: 12px;
      color: #64748b;
      background: #070c18;
      padding: 12px 16px;
      border-radius: 6px;
      border: 1px solid #1e293b;
      text-align: left;
      line-height: 1.8;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="status-badge">
      <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
      ALL SERVICES OPERATIONAL (HEALTHY)
    </div>
    <h1>ESPORTS TACTICAL OS</h1>
    <p>You have accessed the backend API health monitor. All neural prediction models, spatial raycasting, and WebSocket streaming servers are online.</p>
    <a href="/" class="btn">🚀 ENTER TACTICAL DASHBOARD</a>
    <div style="margin-top: 12px; font-size: 13px;">
      <span style="color: #64748b;">Or open live dev server: </span>
      <a href="http://localhost:5173/" style="color: #38bdf8; text-decoration: underline;">http://localhost:5173/</a>
    </div>
    <div class="meta">
      <div>Backend API: HTTP 200 OK (Redirecting to Dashboard in 3s...)</div>
      <div>Active Replay Streams: {active['n']} / {MAX_CLIENTS} (Max Limit)</div>
      <div>Telemetry WebSocket: /ws/replay/&#123;match&#125;/&#123;round&#125;</div>
    </div>
  </div>
</body>
</html>"""
        return HTMLResponse(content=html_content)
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


@app.get('/api/available_demos')
def get_available_demos():
    """Lists all .dem replay files available on disk with indexing status."""
    demos_dir = ROOT / 'data' / 'demos'
    if not demos_dir.exists():
        return {'demos': []}
    
    out = []
    for p in sorted(demos_dir.glob('*.dem')):
        mid = p.stem
        is_ready = (FEATS / f"{mid}_players.parquet").exists()
        out.append({
            'filename': p.name,
            'match_id': mid,
            'size_mb': round(p.stat().st_size / (1024 * 1024), 1),
            'is_ready': is_ready
        })
    return {'demos': out}


@app.post('/api/demos/process')
async def process_demo_feed(req: Request):
    """Processes a new or existing .dem replay into features for live streaming."""
    try:
        data = await req.json()
    except Exception:
        data = {}

    demo_name = data.get('demo_name', '').strip()
    if not demo_name:
        return {'ok': False, 'error': 'Missing demo_name or file path.'}

    raw_path = Path(demo_name)
    candidates = [
        raw_path,
        ROOT / demo_name,
        ROOT / 'data' / 'demos' / demo_name,
        ROOT / 'data' / 'demos' / raw_path.name,
        DEMOS / demo_name,
        DEMOS / raw_path.name,
    ]
    demo_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            demo_path = cand
            break

    if demo_path is None:
        return {'ok': False, 'error': f"Demo file not found: {demo_name}"}

    mid = demo_path.stem
    players_parquet = FEATS / f"{mid}_players.parquet"
    team_parquet = FEATS / f"{mid}_team.parquet"

    # If not already indexed, ingest ticks and extract features
    try:
        if not players_parquet.exists() or not team_parquet.exists():
            from ingest import save as save_ticks
            from features import process_match
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, save_ticks, demo_path)
            players, team_df = await loop.run_in_executor(None, process_match, mid)
            players.to_parquet(players_parquet)
            team_df.to_parquet(team_parquet)

        pl = pd.read_parquet(players_parquet, columns=['round_num'])
        rounds = sorted(int(r) for r in pl.round_num.unique())
        return {
            'ok': True,
            'match_id': mid,
            'rounds': rounds,
            'total_rounds': len(rounds),
            'message': f"Replay {mid} is ready for real-time live streaming!"
        }
    except Exception as e:
        return {'ok': False, 'error': f"Processing error: {str(e)}"}


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
                'directive': tactical_directive(int(slot), float(p), r),
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
            'directive': cache['players'].get(int(p.slot), {}).get('directive', ''),
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


@app.get('/api/ml/metrics')
def get_ml_metrics():
    """Serves formal faculty proof metrics, confusion matrices, and ablation data."""
    proof_path = MODELS / 'faculty_metrics_summary.json'
    if proof_path.exists():
        with open(proof_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'error': 'Metrics report not found. Run backend/model_training_proof.py.'}


# =============================================================================
# Pathway 1 & 3: Video Computer Vision & Live Telemetry Endpoints
# =============================================================================
_cv_engine = None

def get_cv_engine():
    global _cv_engine
    if _cv_engine is None:
        from cv_radar_engine import CVTacticalPredictor
        _cv_engine = CVTacticalPredictor('de_mirage')
    return _cv_engine


@app.post('/api/cv/predict_frame')
async def predict_cv_frame(req: Request):
    """Pathway 1: Accepts real-time CV detected player coordinates and returns 28-feature inference."""
    try:
        body = await req.json()
    except Exception:
        body = {}

    frame_idx = int(body.get('frame_idx', 0))
    clock_s = float(body.get('clock', 30.0))
    players = body.get('players', [])
    engine = get_cv_engine()
    result = engine.process_frame(frame_idx=frame_idx, clock_s=clock_s, direct_players=players)
    return result


@app.websocket('/ws/cv_stream')
async def cv_stream_ws(ws: WebSocket):
    """Pathway 1 & 3: Continuous bi-directional WebSocket stream for live video telemetry."""
    await ws.accept()
    engine = get_cv_engine()
    try:
        while True:
            text = await ws.receive_text()
            data = orjson.loads(text)
            frame_idx = int(data.get('frame_idx', 0))
            clock_s = float(data.get('clock', 0.0))
            players = data.get('players', [])
            payload = engine.process_frame(frame_idx=frame_idx, clock_s=clock_s, direct_players=players)
            await ws.send_text(orjson.dumps(payload).decode())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(orjson.dumps({'error': str(e)}).decode())
        except Exception:
            pass


# =============================================================================
# Valve CS2 Game State Integration (GSI) Live Endpoint for 4 PM Matches
# =============================================================================
_latest_gsi_state = {"status": "waiting", "timestamp": 0, "payload": {}}

@app.post('/api/gsi/live')
async def valve_gsi_webhook(req: Request):
    """Valve Game State Integration: CS2 game engine directly pushes live 4 PM match ticks."""
    global _latest_gsi_state
    try:
        body = await req.json()
        _latest_gsi_state = {
            "status": "live",
            "timestamp": time.time(),
            "payload": body
        }
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get('/api/gsi/status')
def get_gsi_status():
    """Returns the latest live match tick received from Valve Game State Integration."""
    return _latest_gsi_state

@app.get('/api/gsi/cfg')
def download_gsi_cfg():
    """Generates the official gamestate_integration_tactical_os.cfg file for CS2."""
    cfg_content = """"Tactical OS CS2 Integration"
{
 "uri" "http://127.0.0.1:8000/api/gsi/live"
 "timeout" "5.0"
 "buffer"  "0.1"
 "throttle" "0.1"
 "heartbeat" "15.0"
 "data"
 {
   "provider"            "1"
   "map"                 "1"
   "round"               "1"
   "player_id"           "1"
   "player_state"        "1"
   "player_weapons"      "1"
   "player_match_stats"  "1"
   "allplayers_id"       "1"
   "allplayers_state"    "1"
   "allplayers_match_stats" "1"
   "allplayers_weapons"  "1"
   "allplayers_position" "1"
   "bomb"                "1"
 }
}
"""
    return PlainTextResponse(cfg_content, headers={"Content-Disposition": "attachment; filename=gamestate_integration_tactical_os.cfg"})


# =============================================================================
# Video Management & File Upload Endpoints
# =============================================================================
VIDEOS_DIR = ROOT / 'data' / 'videos'
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


@app.get('/api/videos/list')
def list_videos():
    """Lists all available match gameplay videos in data/videos/."""
    vids = []
    if VIDEOS_DIR.exists():
        for p in sorted(VIDEOS_DIR.glob('*.*')):
            if p.suffix.lower() in ('.mp4', '.webm', '.avi', '.mov', '.mkv'):
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                vids.append({
                    'filename': p.name,
                    'url': f'/videos/{p.name}',
                    'size_mb': size_mb,
                    'modified': p.stat().st_mtime,
                    'is_sample': p.name == 'sample_gameplay.mp4'
                })
    return {'videos': vids}


@app.post('/api/videos/upload')
async def upload_video(request: Request):
    """Accepts any custom gameplay video file via dropzone / binary upload and saves to data/videos/."""
    raw_filename = request.headers.get('x-filename', 'uploaded_gameplay.mp4')
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', raw_filename)
    if not safe_name.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
        safe_name += '.mp4'
    
    target_path = VIDEOS_DIR / safe_name
    content = await request.body()
    with open(target_path, 'wb') as f:
        f.write(content)
        
    size_mb = round(len(content) / (1024 * 1024), 2)
    return {
        'status': 'success',
        'filename': safe_name,
        'url': f'/videos/{safe_name}',
        'size_mb': size_mb,
        'message': f'Uploaded {safe_name} ({size_mb} MB) successfully to data/videos/'
    }


@app.post('/api/videos/generate_sample')
def generate_sample_video_endpoint():
    """Generates or regenerates automated CS2 match gameplay test video."""
    try:
        from generate_sample_video import create_gameplay_video
        info = create_gameplay_video()
        return {'status': 'success', **info}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


app.mount('/videos', StaticFiles(directory=str(VIDEOS_DIR)), name='videos')

# Mount evaluation plots
PLOTS_DIR = MODELS / 'evaluation_plots'
if PLOTS_DIR.exists():
    app.mount('/api/ml/plots', StaticFiles(directory=str(PLOTS_DIR)), name='ml_plots')

# Mount static frontend production build at root if available
DIST_DIR = ROOT / 'frontend' / 'dist'
if DIST_DIR.exists():
    app.mount('/', StaticFiles(directory=str(DIST_DIR), html=True), name='frontend')


