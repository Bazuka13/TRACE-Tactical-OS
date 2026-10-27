# backend/stream_new_demo.py
"""
Real-Time Demo Feed & Live Prediction Engine
Usage:
    python backend/stream_new_demo.py [DEMO_PATH] [--round RND] [--speed SPEED]

Examples:
    python backend/stream_new_demo.py data/demos/Match01_morningstar-vs-fengda-m1-mirage.dem --round 7
    python backend/stream_new_demo.py "C:/path/to/my_new_match.dem" --round 1 --speed 2.0
"""
import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure backend directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import ROOT, DEMOS, TICKS, AUX, FEATS, DT
from maps import load_walls, units_per_norm
from server import load_round, predict_frame, compute_xai, get_models


def ensure_demo_processed(demo_path: Path):
    """Checks if .dem is already ingested & features extracted. If not, processes it."""
    mid = demo_path.stem
    players_parquet = FEATS / f"{mid}_players.parquet"
    team_parquet = FEATS / f"{mid}_team.parquet"

    if players_parquet.exists() and team_parquet.exists():
        print(f"[CACHE] Demo features already indexed for: {mid}")
        return mid

    print(f"[INGESTION] Brand new .dem detected: {demo_path.name}")
    print("  -> Step 1/2: Parsing ticks, events & player trajectories at 10Hz...")
    from ingest import save as save_ticks
    save_ticks(demo_path)

    print("  -> Step 2/2: Computing 28 spatial, line-of-sight & kinematics features...")
    from features import process_match
    players, team_df = process_match(mid)
    players.to_parquet(players_parquet)
    team_df.to_parquet(team_parquet)
    print(f"[READY] Ingestion complete: {len(team_df)} frames generated.\n")
    return mid


def stream_realtime(demo_path: Path, round_num: int = None, speed: float = 1.0, max_frames: int = None):
    mid = ensure_demo_processed(demo_path)
    
    # Load rounds
    pl_all = pd.read_parquet(FEATS / f"{mid}_players.parquet", columns=['round_num'])
    available_rounds = sorted(int(r) for r in pl_all.round_num.unique())
    
    if round_num is None:
        round_num = available_rounds[0] if available_rounds else 1
    elif round_num not in available_rounds:
        print(f"[WARN] Round {round_num} not found. Available rounds: {available_rounds[:10]}...")
        round_num = available_rounds[0]

    print(f"\n" + "="*78)
    print(f"  [ESPORTS TACTICAL OS -- REAL-TIME REPLAY STREAMING FEED]")
    print(f"  Match ID:     {mid}")
    print(f"  Round:        {round_num} of {len(available_rounds)}")
    print(f"  Speed Rate:   {speed}x (Target Delay: {DT/speed*1000:.1f}ms/frame)")
    print(f"="*78 + "\n")

    pl, tm = load_round(mid, round_num)
    map_name = pl.map_name.iloc[0]
    grouped_frames = list(pl.groupby('tick'))
    total_frames = len(grouped_frames)

    target_delay = DT / max(0.2, speed)
    site_a = np.array([0.68, 0.38])
    site_b = np.array([0.38, 0.72])

    print(f"[STREAM START] Beginning live 10Hz prediction stream ({total_frames} frames)...\n")
    
    t_round_start = time.time()
    for frame_idx, (tick, g) in enumerate(grouped_frames):
        t_frame_start = time.time()
        if max_frames and frame_idx >= max_frames:
            break

        trow = tm.loc[tick]
        alive, threat, wp, xw = predict_frame(g, trow)

        # Target Site Prediction
        t_pos = g.loc[alive & (g.team == 0), ['pos_x', 'pos_y']].to_numpy()
        if len(t_pos) > 0:
            da = np.linalg.norm(t_pos - site_a, axis=1).min()
            db = np.linalg.norm(t_pos - site_b, axis=1).min()
            site_target = "SITE A" if da < db else "SITE B"
            site_conf = int(np.clip(50 + (abs(db - da) * 200), 50, 95))
        else:
            site_target = "RETENTION"
            site_conf = 50

        # Every 10 frames (~1 second), generate tactical broadcast insight
        if frame_idx % 10 == 0:
            slots = g.slot.to_numpy()[alive]
            xai = compute_xai(g.loc[alive], slots, threat[alive], xw, wp)
            max_threat_idx = threat.argmax()
            max_threat_slot = g.slot.iloc[max_threat_idx]
            max_threat_val = threat[max_threat_idx]
            clock_s = float(trow['round_clock_s'])
            att_n = int(trow['att_alive'])
            def_n = int(trow['def_alive'])

            threat_badge = "HIGH DANGER" if max_threat_val >= 0.6 else ("ELEVATED" if max_threat_val >= 0.3 else "LOW RISK")
            coach_text = xai['players'].get(int(max_threat_slot), {}).get('text', 'Normal pacing.')

            print(f"[TICK {tick:06d} | {clock_s:5.1f}s]  Att (T): {att_n} vs Def (CT): {def_n} | Win Odds: T {wp*100:4.1f}% vs CT {(1-wp)*100:4.1f}%")
            print(f"  -> Target: {site_target} ({site_conf}%) | Max Ambush: P{max_threat_slot+1} at {max_threat_val*100:4.1f}% [{threat_badge}]")
            print(f"  -> Coach AI: {coach_text}")
            print(f"  " + "-"*72)

        # Enforce realistic real-time clock pacing
        elapsed = time.time() - t_frame_start
        sleep_time = max(0.0, target_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    total_time = time.time() - t_round_start
    print(f"\n[STREAM FINISHED] Streamed {min(total_frames, max_frames or total_frames)} ticks in {total_time:.2f}s.")
    print("="*78 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Real-Time CS2 Demo Replay Streamer & Neural Predictor")
    parser.add_argument('demo', nargs='?', default=str(DEMOS / 'Match01_morningstar-vs-fengda-m1-mirage.dem'),
                        help="Path to .dem replay file")
    parser.add_argument('--round', type=int, default=7, help="Round number to stream")
    parser.add_argument('--speed', type=float, default=2.0, help="Playback speed multiplier (e.g. 1.0, 2.0, 4.0)")
    parser.add_argument('--frames', type=int, default=150, help="Max frames to stream (default 150 for test run)")
    args = parser.parse_args()

    demo_p = Path(args.demo)
    if not demo_p.exists():
        # Check in data/demos
        demo_p = DEMOS / args.demo
        if not demo_p.exists():
            print(f"Error: Demo file not found at {args.demo}")
            sys.exit(1)

    stream_realtime(demo_p, round_num=args.round, speed=args.speed, max_frames=args.frames)
