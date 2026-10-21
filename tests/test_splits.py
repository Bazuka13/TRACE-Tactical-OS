import pandas as pd
from config import DATA


def test_no_leakage_between_splits():
    sp = pd.read_csv(DATA / 'splits.csv')
    assert sp.match_id.is_unique, 'a match is listed twice'
    assert set(sp.split) <= {'train', 'val', 'test'}
    # a tournament must live in ONE split only (delete this line if you have a single tournament)
    assert (sp.groupby('tournament').split.nunique() == 1).all(), 'a tournament is in two splits'
