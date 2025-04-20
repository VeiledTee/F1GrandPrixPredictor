#!/usr/bin/env python
# train_model.py
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

pd.set_option("display.max_rows", None)

###############################################################################
# CONFIG
###############################################################################
FEATURE_CSV  = "engineered_features.csv"
MODEL_OUT    = "f1_lapreg_model.pkl"
META_OUT     = "f1_lapreg_meta.pkl"
TARGET_COL   = "race_avg_lap"
ID_COLS      = ["driver", "team", "year", "round", "final_position", "winner"]
RANDOM_SEED  = 42
N_SEARCH_ITER = 50               # ← tweak for more/less exhaustive search
CV_FOLDS      = 5
###############################################################################


def _prep_dataframe(df: pd.DataFrame):
    """Drop ID / leakage cols and mean‑impute."""
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    if TARGET_COL in X.columns:
        X = X.drop(columns=TARGET_COL)
    return X.fillna(X.mean(numeric_only=True))


def load_training_data(path: str = FEATURE_CSV):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise KeyError(f"{TARGET_COL} missing – run feature_engineering.py first.")
    return _prep_dataframe(df), df[TARGET_COL]


# --------------------------------------------------------------------------- #
# MODEL TRAIN + HYPER‑PARAM SEARCH
# --------------------------------------------------------------------------- #
def _hyperparameter_search(X: pd.DataFrame, y):
    """Randomised CV search over key GBR parameters."""
    base = GradientBoostingRegressor(random_state=RANDOM_SEED)

    param_dist = {
        "n_estimators":   np.arange(200, 1100, 100),
        "learning_rate":  np.linspace(0.01, 0.15, 15),
        "max_depth":      np.arange(3, 8),          # shallow trees avoid over‑fit
        "min_samples_split": np.arange(2, 10),
        "min_samples_leaf":  np.arange(1, 8),
        "subsample":      np.linspace(0.6, 1.0, 5),
        "max_features":   ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        cv=CV_FOLDS,
        random_state=RANDOM_SEED,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
        verbose=1,
    ).fit(X, y)

    print("🔍 Best params:", search.best_params_)
    print(f"🔍 Best CV MAE: {-search.best_score_:.3f} s")
    return search.best_estimator_


def train_and_save(X: pd.DataFrame, y):
    reg = _hyperparameter_search(X, y)

    # simple hold‑out report for sanity
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    print(f"📊 Hold‑out MAE = {mean_absolute_error(y_te, y_pred):.3f} s")
    print(f"📈 Hold‑out R²  = {r2_score(y_te, y_pred):.3f}")

    joblib.dump(reg, MODEL_OUT)
    print(f"💾 Model  →  {MODEL_OUT}")

    meta = {
        "feature_names": list(X.columns),
        "feature_means": X.mean(numeric_only=True).to_dict(),
    }
    joblib.dump(meta, META_OUT)
    print(f"💾 Meta   →  {META_OUT}")
    return reg, meta


# --------------------------------------------------------------------------- #
# PREDICTION (unchanged)
# --------------------------------------------------------------------------- #
def _build_feature_frame_from_quali(q, meta, drv_col, lap_col):
    feats = pd.DataFrame(columns=meta["feature_names"])
    if "best_quali_time" in feats.columns:
        feats["best_quali_time"] = pd.to_numeric(q[lap_col], errors="coerce")
    for col in feats.columns:
        feats[col] = feats[col].fillna(meta["feature_means"].get(col, 0.0))
    return feats


def predict_from_quali(model_path, meta_path, quali_csv):
    reg  = joblib.load(model_path)
    meta = joblib.load(meta_path)

    q = pd.read_csv(quali_csv)
    cols    = {c.lower().strip(): c for c in q.columns}
    drv_col = next((cols[k] for k in cols if k in {"driver", "driver_abbreviation", "abbr"}), None)
    lap_col = next((cols[k] for k in cols if k.startswith("qualifying") or "lap" in k), None)
    if drv_col is None or lap_col is None:
        raise ValueError("CSV must contain driver name/abbr and qualifying‑lap seconds.")

    X_inf = _build_feature_frame_from_quali(q, meta, drv_col, lap_col)
    q["PredictedRaceTime (s)"] = reg.predict(X_inf)
    out = q[[drv_col, "PredictedRaceTime (s)"]].sort_values("PredictedRaceTime (s)")

    print("\n🏁 Predicted GP order (quickest → slowest avg lap) 🏁\n")
    print(out.to_string(index=False))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper‑param tuned GBR + optional quali prediction.")
    parser.add_argument("--quali_csv", type=str, help="Minimal quali CSV for prediction", default=None)
    args = parser.parse_args()

    X, y = load_training_data()
    _, _ = train_and_save(X, y)

    predict_from_quali(MODEL_OUT, META_OUT, 'qualifying_bahrian.csv')
