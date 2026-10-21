# backend/generate_mock_data.py
"""
Generates synthetic CS2 match data that strictly adheres to the schema and
exit criteria of Phase 0, enabling full end-to-end testing, feature generation,
and unit testing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from config import TICKS, AUX, STEP, TICK_RATE
from maps import units_to_norm

def create_synthetic_match(match_id='match01', map_name='de_mirage', n_rounds=12):
    rng = np.random.default_rng(int(hash(match_id) % 100000))
    player_ids = [76561198000000000 + i for i in range(10)]
    
    tick_records = []
    hurt_records = []
    smoke_records = []
    round_records = []
    
    current_tick = 1000
    
    for rnd in range(1, n_rounds + 1):
        start_tick = current_tick
        n_frames = 120  # ~11.25 seconds per round
        round_ticks = np.arange(start_tick, start_tick + n_frames * STEP, STEP)
        end_tick = int(round_ticks[-1])
        winner_team = int(rng.choice([0, 1]))
        plant_tick = int(start_tick + 50 * STEP) if rng.random() > 0.4 else np.nan
        
        round_records.append({
            'start_tick': start_tick,
            'end_tick': end_tick,
            'winner_team': winner_team,
            'round_num': rnd,
            'plant_tick': plant_tick
        })
        
        # Base coordinates for 10 players: slots 0-4 are Team 0 (T), slots 5-9 are Team 1 (CT)
        t_base = np.array([0.25, 0.45])
        ct_base = np.array([0.75, 0.55])
        
        # Player states
        alive = np.ones(10, dtype=bool)
        hp = np.ones(10, dtype=float)
        armor = np.full(10, 0.9, dtype=float)
        
        # An ambush fight scheduled at frame 70: slot 7 (CT) ambushes slot 1 (T) from behind
        ambush_frame = 70
        ambush_tick = int(round_ticks[ambush_frame])
        
        for f_idx, tick in enumerate(round_ticks):
            clock_s = (tick - start_tick) / TICK_RATE
            bomb_planted = int(not np.isnan(plant_tick) and tick >= plant_tick)
            
            # Update ambush hit
            if f_idx == ambush_frame + 2:
                hp[1] = 0.1  # lost 90 HP
                hurt_records.append({
                    'tick': tick,
                    'att_slot': 7,
                    'vic_slot': 1,
                    'dmg_health': 90
                })
            
            for s in range(10):
                team = 0 if s < 5 else 1
                base = t_base if team == 0 else ct_base
                offset = np.array([(s % 5) * 0.05, ((s % 5) % 2) * 0.06])
                drift = np.array([
                    0.08 * np.sin(f_idx / 20.0 + s),
                    0.06 * np.cos(f_idx / 20.0 + s)
                ])
                
                # Attacker slot 7 flanks behind victim slot 1 around ambush_frame
                if s == 7 and f_idx >= ambush_frame - 15:
                    pos = np.array([0.38, 0.48])
                    yaw = 180.0
                elif s == 1 and f_idx >= ambush_frame - 15:
                    pos = np.array([0.28, 0.48])
                    yaw = 0.0  # looking right (away from slot 7)
                else:
                    pos = np.clip(base + offset + drift, 0.05, 0.95)
                    yaw = float((s * 36 + f_idx * 2) % 360)
                
                is_alive_bool = bool(alive[s])
                hp_val = float(hp[s])
                
                # Weapon ratings
                weapon_dps_val = 0.85 if s in (0, 5) else (0.80 if s in (1, 6) else 0.55)
                
                tick_records.append({
                    'game': 'cs2',
                    'match_id': match_id,
                    'map_name': map_name,
                    'round_num': rnd,
                    'tick': int(tick),
                    'round_clock_s': float(clock_s),
                    'player_id': int(player_ids[s]),
                    'slot': int(s),
                    'team': int(team),
                    'pos_x': float(pos[0]),
                    'pos_y': float(pos[1]),
                    'yaw': float(yaw),
                    'hp_pct': float(hp_val),
                    'armor_pct': float(armor[s]),
                    'weapon_dps': float(weapon_dps_val),
                    'is_alive': is_alive_bool,
                    'cash': 4500.0,
                    'bomb_planted': int(bomb_planted),
                    'round_winner': int(winner_team)
                })
        
        # Smoke grenade in the round
        smoke_records.append({
            'tick_start': int(start_tick + 20 * STEP),
            'tick_end': int(start_tick + 60 * STEP),
            'x': 0.50,
            'y': 0.50,
            'r': units_to_norm(144, map_name)
        })
        
        current_tick = end_tick + 100
        
    df = pd.DataFrame(tick_records)
    
    # Calculate team power index
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
    df = df.drop(columns=['alive_i', 'armor_alive', 'wdps_alive'])
    df = df.sort_values(['round_num', 'tick', 'slot']).reset_index(drop=True)
    
    smokes_df = pd.DataFrame(smoke_records)
    hurt_df = pd.DataFrame(hurt_records)
    rounds_df = pd.DataFrame(round_records)
    
    return df, smokes_df, hurt_df, rounds_df

def generate_all_matches():
    sp = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'splits.csv')
    for mid in sp.match_id.unique():
        df, smokes, hurt, rounds = create_synthetic_match(mid)
        df.to_parquet(TICKS / (mid + '.parquet'))
        smokes.to_parquet(AUX / (mid + '_smokes.parquet'))
        hurt.to_parquet(AUX / (mid + '_hurt.parquet'))
        rounds.to_parquet(AUX / (mid + '_rounds.parquet'))
        print(f"Generated synthetic {mid}: {len(df)} ticks, {len(rounds)} rounds, {len(hurt)} hurt events")

if __name__ == '__main__':
    generate_all_matches()
