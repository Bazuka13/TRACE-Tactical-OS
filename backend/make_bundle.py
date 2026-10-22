# backend/make_bundle.py   run:  python backend/make_bundle.py match01 match02 match03
import sys
import shutil
from pathlib import Path
from config import ROOT, FEATS, EMB, MAPS, MODELS

out = ROOT / 'deploy_data'
for sub in ('data/demos', 'data/ticks/aux', 'data/features', 'data/labels',
            'data/embeddings', 'data/maps', 'models'):
    (out / sub).mkdir(parents=True, exist_ok=True)

# Collect matches to bundle
match_ids = sys.argv[1:]
if not match_ids:
    # Default to first 3 matches if none specified
    match_ids = ['Match01_morningstar-vs-fengda-m1-mirage',
                 'Match02_legacy-vs-parivision-m1-mirage',
                 'Match03_detonate-vs-kodex-m1-mirage']

for m in match_ids:
    m_clean = m.replace('_players.parquet', '').replace('.parquet', '')
    for src, dst in [(FEATS / (m_clean + '_players.parquet'), 'data/features'),
                     (FEATS / (m_clean + '_team.parquet'), 'data/features'),
                     (EMB / (m_clean + '_gnn.parquet'), 'data/embeddings')]:
        if src.exists():
            shutil.copy(src, out / dst)
            print(f'Copied {src.name} -> {dst}')

for f in MAPS.glob('*_walls.json'):
    shutil.copy(f, out / 'data/maps')
    print(f'Copied wall map {f.name} -> data/maps')

for f in MODELS.glob('*'):
    if f.name not in ('stgnn.pt', 'TEST_ALREADY_USED'):
        shutil.copy(f, out / 'models')
        print(f'Copied model {f.name} -> models')

print(f'\nDeployment bundle successfully prepared in {out}')
