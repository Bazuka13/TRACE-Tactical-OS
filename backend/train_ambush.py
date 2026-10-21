# backend/train_ambush.py run: python backend/train_ambush.py ablate
# python backend/train_ambush.py final T4 (later: final T5)
import sys
import json
import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from config import MODELS, FEATURES, GNN_COLS
from dataset import load_labeled

T1 = ['hp_pct', 'armor_pct', 'weapon_dps', 'is_alive', 'team_power_idx',
      'round_clock_norm', 'round_phase_enc']
T2 = T1 + ['pos_x', 'pos_y', 'yaw_sin', 'yaw_cos', 'territory_control_pct',
           'is_in_enemy_territory', 'min_enemy_dist']
T3 = T2 + ['vel_x', 'vel_y', 'speed', 'accel_mag', 'speed_1s_mean', 'speed_3s_mean']
T4 = T3 + ['n_enemies_visible', 'n_teammates_visible', 'max_enemy_los_dist',
           'nearest_danger_score', 'degree', 'is_isolated', 'has_crossfire_partner',
           'anchor_influence_score']

assert sorted(T4) == sorted(FEATURES)
TIERS = {
    'T1 economy': T1,
    'T2 + space': T2,
    'T3 + movement': T3,
    'T4 + sight and graph': T4,
    'T5 + neural net': T4 + GNN_COLS
}


def fit(cols, tr, va):
    n_pos = int(tr.y_ambush.sum())
    n_neg = len(tr) - n_pos
    scale = (n_neg / n_pos) if n_pos > 0 else 1.0
    m = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6,
                           loss_function='Logloss', eval_metric='AUC',
                           scale_pos_weight=scale, early_stopping_rounds=100,
                           random_seed=42, verbose=0)
    m.fit(tr[cols], tr.y_ambush, eval_set=(va[cols], va.y_ambush))
    return m


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python backend/train_ambush.py ablate | final <T4|T5>")
        sys.exit(1)

    mode = sys.argv[1]
    tr, va = load_labeled('train'), load_labeled('val')
    print('train rows', len(tr), 'positives', int(tr.y_ambush.sum()), '| val rows', len(va))

    try:
        tr5, va5 = load_labeled('train', True), load_labeled('val', True)
    except Exception:
        tr5 = va5 = None  # neural-net embeddings do not exist yet

    if mode == 'ablate':
        print('%-24s %s' % ('T0 random guessing', 'PR-AUC %.3f' % va.y_ambush.mean()))
        for name, cols in TIERS.items():
            a, b = (tr5, va5) if name.startswith('T5') else (tr, va)
            if a is None:
                print('%-24s (run the neural net first)' % name)
                continue
            m = fit(cols, a, b)
            p = m.predict_proba(b[cols])[:, 1]
            print('%-24s PR-AUC %.3f' % (name, average_precision_score(b.y_ambush, p)))

    elif mode == 'final':
        tier = sys.argv[2] if len(sys.argv) > 2 else 'T4'
        name = [k for k in TIERS if k.startswith(tier)][0]
        cols = TIERS[name]
        a, b = (tr5, va5) if tier == 'T5' else (tr, va)
        m = fit(cols, a, b)
        raw = m.predict_proba(b[cols])[:, 1]
        cal = IsotonicRegression(out_of_bounds='clip').fit(raw, b.y_ambush)  # fitted on VAL only
        m.save_model(str(MODELS / 'ambush_cb.cbm'))
        joblib.dump(cal, MODELS / 'ambush_cal.joblib')
        with open(MODELS / 'ambush_cols.json', 'w') as f:
            json.dump(cols, f)
        print('saved', name, '| val PR-AUC %.3f' % average_precision_score(b.y_ambush, raw))
