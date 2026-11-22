"""Load, clean, engineer features and split the weather-anomaly data (chronological split)."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ART, CLASSES, NUM_FEATURES, SEED, TRAIN_END_YEAR, VAL_END_YEAR

# NOTE: `degrees_from_mean` is deliberately NOT used. The class label is derived from it
# (a tree model reaches ~99.6% with it), so using it would leak the answer.


def load_clean(path):
    cols = ["date_str", "id", "longitude", "latitude", "max_temp", "min_temp", "type"]
    df = pd.read_csv(path, usecols=cols, parse_dates=["date_str"])
    n_raw = len(df)
    ok = (df["type"].isin(CLASSES) & (df.min_temp <= df.max_temp)
          & df.max_temp.between(-60, 60) & df.min_temp.between(-60, 60)
          & df.latitude.between(-90, 90) & df.longitude.between(-180, 180))
    return df[ok].reset_index(drop=True), n_raw


def make_features(max_t, min_t, lat, lon, doy):
    max_t, min_t, lat, lon, doy = (np.asarray(v, dtype="float64") for v in (max_t, min_t, lat, lon, doy))
    return np.column_stack([max_t, min_t, max_t - min_t, (max_t + min_t) / 2, lat, lon,
                            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]).astype("float32")


def build_datasets(path, max_train=600_000, max_val=100_000, seed=SEED, save=True):
    df, n_raw = load_clean(path)
    yr = df.date_str.dt.year.values
    parts = {"train": yr <= TRAIN_END_YEAR, "val": (yr > TRAIN_END_YEAR) & (yr <= VAL_END_YEAR), "test": yr > VAL_END_YEAR}
    caps = {"train": max_train, "val": max_val, "test": None}
    sets, split_info = {}, {}
    for k, m in parts.items():
        d = df[m]
        used = d.sample(caps[k], random_state=seed) if caps[k] and len(d) > caps[k] else d
        sets[k] = used
        split_info[k] = {"years": f"{int(d.date_str.dt.year.min())}-{int(d.date_str.dt.year.max())}",
                         "rows_total": int(len(d)), "rows_used": int(len(used))}
    stations = sorted(df.loc[parts["train"], "id"].unique())
    st_index = {s: i + 1 for i, s in enumerate(stations)}                 # 0 = unseen station
    feat = {k: make_features(d.max_temp, d.min_temp, d.latitude, d.longitude, d.date_str.dt.dayofyear) for k, d in sets.items()}
    scaler = StandardScaler().fit(feat["train"])
    cls = {c: i for i, c in enumerate(CLASSES)}
    out = {k: {"X": scaler.transform(feat[k]).astype("float32"),
               "s": d["id"].map(st_index).fillna(0).astype("int32").values,
               "y": d["type"].map(cls).astype("int32").values} for k, d in sets.items()}
    info = {"rows_raw": int(n_raw), "rows_clean": int(len(df)), "rows_removed": int(n_raw - len(df)),
            "stations": int(df["id"].nunique()), "stations_in_train": len(stations),
            "date_min": str(df.date_str.min().date()), "date_max": str(df.date_str.max().date()),
            "split": split_info, "features": NUM_FEATURES + ["station embedding"],
            "class_counts": {k: np.bincount(out[k]["y"], minlength=len(CLASSES)).tolist() for k in out}}
    if save:
        joblib.dump(scaler, ART / "scaler.joblib")
        (ART / "stations.json").write_text(json.dumps(st_index))
    return out, info, len(stations)
