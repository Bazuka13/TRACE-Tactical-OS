# backend/explain.py test: python backend/explain.py
import json
import joblib
import numpy as np
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier
from config import MODELS

# technical name -> words a coach would use. Show THESE on the dashboard, never raw feature names.
HUMAN = {
    'pos_x': 'Position on the map',
    'pos_y': 'Position on the map',
    'yaw_sin': 'Direction they are facing',
    'yaw_cos': 'Direction they are facing',
    'hp_pct': 'Health',
    'armor_pct': 'Armor',
    'weapon_dps': 'Weapon strength',
    'is_alive': 'Alive',
    'vel_x': 'Movement direction',
    'vel_y': 'Movement direction',
    'speed': 'Current running speed',
    'accel_mag': 'Sudden change of speed',
    'speed_1s_mean': 'Speed over the last second',
    'speed_3s_mean': 'Sprinting forward over the last 3 seconds',
    'n_enemies_visible': 'Enemies in direct sightline',
    'n_teammates_visible': 'Teammates in sightline',
    'min_enemy_dist': 'Distance to the nearest enemy',
    'max_enemy_los_dist': 'Distance to the farthest visible enemy',
    'territory_control_pct': 'Share of the map their team controls',
    'nearest_danger_score': 'Hidden enemies close by',
    'is_in_enemy_territory': 'Closer to enemies than to teammates',
    'degree': 'Teammates close enough to help',
    'is_isolated': 'No teammate nearby to trade',
    'has_crossfire_partner': 'A teammate covering another angle',
    'anchor_influence_score': 'Firepower of nearby teammates',
    'team_power_idx': 'Team money, armor and weapons',
    'round_clock_norm': 'Time left in the round',
    'round_phase_enc': 'Phase of the round',
    'gnn': 'Overall team movement pattern (neural net)',
    'att_power': 'Attack team money, armor and weapons',
    'def_power': 'Defense team money, armor and weapons',
    'power_delta': 'Equipment advantage',
    'att_alive': 'Attackers alive',
    'def_alive': 'Defenders alive',
    'alive_diff': 'Player-count advantage',
    'bomb_planted': 'Bomb planted',
    'att_territory': 'Share of the map controlled',
    'centroid_speed': 'Speed of the attack push',
    'att_disp': 'How spread out the attackers are',
    'def_disp': 'How spread out the defenders are',
}


def load_models():
    return dict(
        cb=CatBoostClassifier().load_model(str(MODELS / 'ambush_cb.cbm')),
        cb_cal=joblib.load(MODELS / 'ambush_cal.joblib'),
        a_cols=json.load(open(MODELS / 'ambush_cols.json')),
        lg=lgb.Booster(model_file=str(MODELS / 'win_lgb.txt')),
        w_cal=joblib.load(MODELS / 'win_cal.joblib'),
        w_cols=json.load(open(MODELS / 'win_cols.json'))
    )


def shap_array(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # some library versions return one array per class
        sv = sv[-1]
    return np.asarray(sv)  # shape: (rows, features)


def top_reasons(shap_row, names, k=3):
    agg = {}
    for n, s in zip(names, shap_row):
        key = 'gnn' if (n[0] == 'g' and n[1:].isdigit()) else n
        text = HUMAN.get(key, key)
        agg[text] = agg.get(text, 0.0) + float(s)  # merge features that mean the same to a human
    total = sum(abs(v) for v in agg.values()) + 1e-9
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [{'text': t, 'up': v > 0, 'share': int(round(100 * abs(v) / total))} for t, v in top]


class Explainers:
    def __init__(self, M):
        self.M = M
        self.cb = shap.TreeExplainer(M['cb'])
        self.lg = shap.TreeExplainer(M['lg'])

    def players(self, X):  # X: table of living players -> one list of 3 reasons each
        cols = self.M['a_cols']
        return [top_reasons(row, cols) for row in shap_array(self.cb, X[cols])]

    def win(self, X):  # X: one row of team features
        cols = self.M['w_cols']
        return top_reasons(shap_array(self.lg, X[cols])[0], cols)


def threat_sentence(slot, p, reasons):
    level = 'HIGH' if p >= 0.6 else 'MEDIUM' if p >= 0.3 else 'LOW'
    parts = []
    for i, r in enumerate(reasons, 1):
        verb = 'raises' if r['up'] else 'lowers'
        parts.append('%d) %s (%s the risk, %d%%)' % (i, r['text'], verb, r['share']))
    return 'Player %d danger %s because: %s' % (slot + 1, level, '; '.join(parts))


def win_sentence(wp, reasons):
    parts = []
    for i, r in enumerate(reasons, 1):
        side = 'favours attack' if r['up'] else 'favours defense'
        parts.append('%d) %s (%s, %d%%)' % (i, r['text'], side, r['share']))
    return 'Attack win chance %d%% because: %s' % (round(100 * wp), '; '.join(parts))


if __name__ == '__main__':
    from dataset import load_labeled
    M = load_models()
    E = Explainers(M)
    uses_gnn = any(c[0] == 'g' and c[1:].isdigit() for c in M['a_cols'])
    rows = load_labeled('val', with_gnn=uses_gnn)
    rows = rows[rows.y_ambush == 1].head(10)
    probs = M['cb_cal'].transform(M['cb'].predict_proba(rows[M['a_cols']])[:, 1])
    for slot, p, r in zip(rows.slot, probs, E.players(rows)):
        print(threat_sentence(int(slot), p, r))
