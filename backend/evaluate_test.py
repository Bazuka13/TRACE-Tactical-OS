# backend/evaluate_test.py run ONCE, after you are completely done tuning
import sys
import json
import joblib
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from config import MODELS
from dataset import load_labeled, load_team, ece

flag = MODELS / 'TEST_ALREADY_USED'
if flag.exists() and '--force' not in sys.argv:
    sys.exit('You already looked at the test set. Tuning on it again makes it useless.')

cols = json.load(open(MODELS / 'ambush_cols.json'))
uses_gnn = any(c[0] == 'g' and c[1:].isdigit() for c in cols)
te = load_labeled('test', with_gnn=uses_gnn)
cb = CatBoostClassifier().load_model(str(MODELS / 'ambush_cb.cbm'))
p = joblib.load(MODELS / 'ambush_cal.joblib').transform(cb.predict_proba(te[cols])[:, 1])
print('AMBUSH PR-AUC %.3f (random = %.3f) Brier %.4f' % (
    average_precision_score(te.y_ambush, p), te.y_ambush.mean(),
    brier_score_loss(te.y_ambush, p)))

wc = json.load(open(MODELS / 'win_cols.json'))
tw = load_team('test')
lg = lgb.Booster(model_file=str(MODELS / 'win_lgb.txt'))
pw = np.clip(joblib.load(MODELS / 'win_cal.joblib').transform(lg.predict(tw[wc])), 1e-4, 1 - 1e-4)
print('WIN   log-loss %.3f Brier %.4f ECE %.4f (target below 0.025)' % (
    log_loss(tw.y_win, pw), brier_score_loss(tw.y_win, pw), ece(tw.y_win, pw)))

flag.write_text('done')
