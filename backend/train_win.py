# backend/train_win.py run: python backend/train_win.py
import json
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from config import MODELS
from dataset import load_team, ece

W1 = ['att_power', 'def_power', 'power_delta']
W2 = W1 + ['att_alive', 'def_alive', 'alive_diff', 'round_clock_norm', 'bomb_planted']
W3 = W2 + ['att_territory', 'centroid_speed', 'att_disp', 'def_disp']

params = dict(
    objective='binary',
    metric=['binary_logloss', 'auc'],
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=200,  # frames in one round are near-copies: demand lots of evidence
    seed=42,
    verbose=-1
)


def train(cols, tr, va):
    return lgb.train(
        params,
        lgb.Dataset(tr[cols], tr.y_win),
        800,
        valid_sets=[lgb.Dataset(va[cols], va.y_win)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )


if __name__ == '__main__':
    tr, va = load_team('train'), load_team('val')
    print('baseline log-loss (always say 50%%): 0.693')
    for name, cols in (('W1 economy', W1), ('W2 + alive, time, bomb', W2), ('W3 + space, movement', W3)):
        m = train(cols, tr, va)
        p = np.clip(m.predict(va[cols]), 1e-4, 1 - 1e-4)
        print('%-24s log-loss %.3f Brier %.3f ECE %.3f' % (
            name, log_loss(va.y_win, p), brier_score_loss(va.y_win, p), ece(va.y_win, p)))

    cols = W3
    m = train(cols, tr, va)
    raw = m.predict(va[cols])
    cal = IsotonicRegression(out_of_bounds='clip').fit(raw, va.y_win)  # fitted on VAL only
    p = np.clip(cal.transform(raw), 1e-4, 1 - 1e-4)
    print('after calibration (val): log-loss %.3f ECE %.3f' % (log_loss(va.y_win, p), ece(va.y_win, p)))

    m.save_model(str(MODELS / 'win_lgb.txt'))
    joblib.dump(cal, MODELS / 'win_cal.joblib')
    with open(MODELS / 'win_cols.json', 'w') as f:
        json.dump(cols, f)
