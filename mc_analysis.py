from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_paired_data():
    def load(path, name):
        d = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["Date/Time"])
        return d.set_index("Date/Time")[["Mean Temp (°C)"]].rename(columns={"Mean Temp (°C)": name})
    frd = load(DATA_DIR / "en_climate_daily_NB_8101505_2019_P1D.csv", "FRD_T")
    stj = load(DATA_DIR / "en_climate_daily_NB_8104901_2019_P1D.csv", "STJ_T")
    return pd.concat([frd, stj], axis=1).loc["2019-05-01":"2019-08-31"].dropna()

def make_lagged_pairs(data, lag_days=0):
    # Positive lag: pair Fredericton(t) with Saint John(t-lag).
    out = data.copy()
    out["STJ_lagged"] = out["STJ_T"].shift(lag_days)
    return out[["FRD_T", "STJ_lagged"]].dropna().rename(columns={"STJ_lagged":"STJ_T"})

def fit_ols(sample, intercept=True):
    x = sample["STJ_T"].to_numpy(dtype=float)
    y = sample["FRD_T"].to_numpy(dtype=float)
    X = sm.add_constant(x) if intercept else x
    return sm.OLS(y, X).fit()

def draw_iid_indices(n_rows, sample_size, rng):
    return rng.integers(0, n_rows, size=sample_size)

def draw_circular_block_indices(n_rows, sample_size, block_length, rng):
    # Circular moving-block bootstrap: concatenate randomly chosen contiguous blocks.
    block_length = max(1, min(int(block_length), n_rows))
    starts = rng.integers(0, n_rows, size=int(np.ceil(sample_size / block_length)))
    idx = np.concatenate([(s + np.arange(block_length)) % n_rows for s in starts])
    return idx[:sample_size]

def monte_carlo(data, sample_size=30, repetitions=500, block_length=5,
                scheme="IID day resampling", intercept=True, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 3:
        raise ValueError("At least three complete paired days are required.")
    sample_size = max(3, min(int(sample_size), n))
    slopes, intercepts, r2s = [], [], []
    for _ in range(int(repetitions)):
        if scheme == "IID day resampling":
            idx = draw_iid_indices(n, sample_size, rng)
        else:
            idx = draw_circular_block_indices(n, sample_size, block_length, rng)
        sample = data.iloc[idx]
        try:
            model = fit_ols(sample, intercept=intercept)
            slopes.append(float(np.asarray(model.params)[-1]))
            intercepts.append(float(np.asarray(model.params)[0]) if intercept else 0.0)
            r2s.append(float(model.rsquared))
        except Exception:
            continue
    return pd.DataFrame({"slope":slopes, "intercept":intercepts, "r_squared":r2s})
