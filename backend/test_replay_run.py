# backend/test_replay_run.py
import time
import numpy as np
import pandas as pd
from server import load_round, predict_frame, compute_xai
from maps import load_walls, units_per_norm

def test_replay(mid='Match01_morningstar-vs-fengda-m1-mirage', rnd=7):
    print(f"=== TESTING REAL REPLAY: {mid} (Round {rnd}) ===")

    pl, tm = load_round(mid, rnd)
    map_name = pl.map_name.iloc[0]
    walls = load_walls(map_name)

    frames = list(pl.groupby('tick'))
    print(f"Total frames in round: {len(frames)}")

    ambush_tick = 57868
    sample_outputs = []

    t_start = time.time()
    n_tested = 0

    for i, (tick, g) in enumerate(frames):
        trow = tm.loc[tick]
        alive, threat, wp, xw = predict_frame(g, trow)
        n_tested += 1

        # Sample snapshots across the round: early round, ambush moment, and late round
        is_ambush = abs(tick - ambush_tick) <= 12
        is_early = (i == 10)
        is_late = (i == len(frames) - 20)

        if is_ambush or is_early or is_late:
            slots = g.slot.to_numpy()[alive]
            xai = compute_xai(g.loc[alive], slots, threat[alive], xw, wp)
            max_threat_idx = threat.argmax()
            max_threat_slot = g.slot.iloc[max_threat_idx]
            max_threat_val = threat[max_threat_idx]

            tag = "AMBUSH MOMENT" if is_ambush else ("EARLY ROUND" if is_early else "LATE ROUND / POST-PLANT")
            sample_outputs.append({
                'tag': tag,
                'tick': int(tick),
                'clock': round(float(trow['round_clock_s']), 1),
                'att_alive': int(trow['att_alive']),
                'def_alive': int(trow['def_alive']),
                'win_prob': round(wp * 100, 1),
                'max_threat_slot': int(max_threat_slot),
                'max_threat_val': round(float(max_threat_val), 3),
                'threats': [round(float(t), 2) for t in threat],
                'why': xai['players'].get(int(max_threat_slot), {}).get('text', ''),
                'why_win': xai['win']['text']
            })

    elapsed = time.time() - t_start
    ms_per_frame = (elapsed / n_tested) * 1000

    print(f"\n[PERFORMANCE] Processed {n_tested} frames in {elapsed:.2f}s ({ms_per_frame:.2f} ms/frame)")
    print(f"[THROUGHPUT] Can simulate {1000/ms_per_frame:.1f} FPS (Target: 10.7 FPS -> {(1000/ms_per_frame)/10.7:.1f}x real-time!)\n")

    print("=== LIVE REPLAY PREDICTION TIMELINE ===")
    for s in sample_outputs:
        print(f"[{s['tag']}] Tick {s['tick']} | Clock: {s['clock']}s | T: {s['att_alive']} vs CT: {s['def_alive']} | Attack Win Prob: {s['win_prob']}%")
        print(f"  Danger per player (slots 1-10): {s['threats']}")
        print(f"  -> Highest Danger: Player {s['max_threat_slot']+1} ({s['max_threat_val']*100:.1f}%)")
        print(f"  -> Coach Danger Insight: {s['why']}")
        print(f"  -> Coach Win Prob Insight: {s['why_win']}\n")

if __name__ == '__main__':
    test_replay()
