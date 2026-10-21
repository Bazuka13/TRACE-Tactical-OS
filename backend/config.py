# backend/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DEMOS = DATA / 'demos'
TICKS = DATA / 'ticks'
AUX = TICKS / 'aux'  # smokes, damage events, round tables
FEATS = DATA / 'features'
LABELS = DATA / 'labels'
EMB = DATA / 'embeddings'
MAPS = DATA / 'maps'
MODELS = ROOT / 'models'

for p in (DEMOS, TICKS, AUX, FEATS, LABELS, EMB, MAPS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

TICK_RATE = 64  # CS2 demo ticks per second
STEP = 6  # keep every 6th tick
DT = STEP / TICK_RATE  # real seconds between kept frames (0.09375)
ROUND_SECONDS = 115.0

# The 28 tabular features, in a fixed order (matches the MVP guide table)
FEATURES = [
    # spatial state (8)
    'pos_x', 'pos_y', 'yaw_sin', 'yaw_cos', 'hp_pct', 'armor_pct',
    'weapon_dps', 'is_alive',
    # kinematics (6)
    'vel_x', 'vel_y', 'speed', 'accel_mag', 'speed_1s_mean', 'speed_3s_mean',
    # line of sight / threat (4)
    'n_enemies_visible', 'n_teammates_visible', 'min_enemy_dist',
    'max_enemy_los_dist',
    # territory (3)
    'territory_control_pct', 'nearest_danger_score', 'is_in_enemy_territory',
    # support graph (4)
    'degree', 'is_isolated', 'has_crossfire_partner',
    'anchor_influence_score',
    # economy and round (3)
    'team_power_idx', 'round_clock_norm', 'round_phase_enc',
]
assert len(FEATURES) == 28
GNN_COLS = ['g%d' % i for i in range(128)]
