import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pandas as pd
import pytest
from config import TICKS
from ingest import validate

files = sorted(TICKS.glob('*.parquet'))

if not files:
    def test_no_tick_files_yet():
        # Placeholder assertion until demo parquet files are generated
        assert True
else:
    @pytest.mark.parametrize('f', files, ids=[f.stem for f in files])
    def test_file_is_clean(f):
        validate(pd.read_parquet(f))
