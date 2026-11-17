from pathlib import Path

import pandas as pd

from synthrep.data import load_htru2


def test_htru2_headerless_loader_keeps_first_row(tmp_path: Path):
    rows = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0],
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1],
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "HTRU_2.csv", index=False, header=False)
    bundle = load_htru2(tmp_path)
    assert bundle.X.shape == (2, 8)
    assert bundle.y.tolist() == [0, 1]
    assert bundle.X.iloc[0, 0] == 1.0
    assert "profile_excess_kurtosis" in bundle.X.columns
