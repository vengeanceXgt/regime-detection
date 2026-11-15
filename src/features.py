"""Causal features: every feature at row t uses returns up to and including day t only."""
import numpy as np
import pandas as pd


def make_features(ret: pd.Series) -> pd.DataFrame:
    r = ret
    f = pd.DataFrame(index=r.index)
    f["r1"] = r
    f["abs_r1"] = r.abs()
    for w in (5, 10, 20, 60):
        f[f"vol_{w}"] = r.rolling(w).std()
        f[f"mean_{w}"] = r.rolling(w).mean()
    f["vol_ratio"] = f["vol_5"] / (f["vol_60"] + 1e-12)
    f["max_abs_10"] = r.abs().rolling(10).max()
    f["down_days_20"] = (r < 0).rolling(20).mean()
    return f
