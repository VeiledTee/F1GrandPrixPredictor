import argparse, joblib, warnings, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split,
)
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

###############################################################################
# CONFIG
###############################################################################
FEATURE_CSV = "engineered_features.csv"
TARGET_COL = "position_delta"  # <--- now predicting position change
ID_COLS = ["driver", "team", "year", "round", "final_position", "winner"]
RANDOM_SEED = 42
CV_FOLDS = 5
N_ITER_GBR = 50
N_ITER_MLP = 40
###############################################################################


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _prep_dataframe(df: pd.DataFrame):
    df = df.copy()
    if "start_grid" not in df.columns or "final_position" not in df.columns:
        raise KeyError(
            "Both 'start_grid' and 'final_position' must be in your dataset."
        )
    df["position_delta"] = (
        df["start_grid"] - df["final_position"]
    )  # positive = gained places
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    y = df["position_delta"]
    X = X.drop(columns=["position_delta"], errors="ignore").fillna(
        X.mean(numeric_only=True)
    )
    return X, y


def load_training_data(path=FEATURE_CSV):
    df = pd.read_csv(path)
    return _prep_dataframe(df)


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def build_gbr():
    base = GradientBoostingRegressor(random_state=RANDOM_SEED)
    param_dist = {
        "n_estimators": np.arange(100, 2600, 100),
        "learning_rate": np.linspace(0.01, 0.15, 50),
        "max_depth": np.arange(3, 11),
        "min_samples_split": np.arange(2, 16),
        "min_samples_leaf": np.arange(1, 11),
        "subsample": np.linspace(0.6, 1.0, 20),
        "max_features": ["sqrt", "log2", None],
    }
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=N_ITER_GBR,
        cv=CV_FOLDS,
        random_state=RANDOM_SEED,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
        verbose=0,
    )


def build_lasso():
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(random_state=RANDOM_SEED, max_iter=5000)),
        ]
    )
    alphas = {"lasso__alpha": np.logspace(-4, 1, 100)}
    return GridSearchCV(
        pipe,
        param_grid=alphas,
        cv=CV_FOLDS,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
    )


def build_mlp():
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    random_state=RANDOM_SEED,
                    max_iter=2000,
                    solver="adam",
                    early_stopping=True,
                ),
            ),
        ]
    )
    param_dist = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (64, 32)],
        "mlp__alpha": np.logspace(-5, -2, 50),
        "mlp__learning_rate_init": np.linspace(0.0005, 0.01, 20),
    }
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_MLP,
        cv=CV_FOLDS,
        random_state=RANDOM_SEED,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
        verbose=0,
    )


# --------------------------------------------------------------------------- #
# Train‑evaluate‑save loop
# --------------------------------------------------------------------------- #
def fit_and_report(name, model, X_tr, X_te, y_tr, y_te, feature_names, means):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae, r2 = mean_absolute_error(y_te, y_pred), r2_score(y_te, y_pred)
    print(f"{name:<9} | MAE = {mae:6.3f} delta | R² = {r2:6.3f}")

    out_path = f"{name.lower()}_model.pkl"
    meta_out = f"{name.lower()}_meta.pkl"
    joblib.dump(model, out_path)
    joblib.dump({"feature_names": feature_names, "feature_means": means}, meta_out)
    return name, mae, r2, out_path


def main(quali_csv: str = ""):
    X, y = load_training_data()
    print("Training target mean:", y.mean())
    print("Target std deviation:", y.std())

    means = X.mean(numeric_only=True).to_dict()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Model      |  Test set metrics")
    print("-" * 40)
    results = []
    for name, builder in [
        ("GBR", build_gbr),
        ("Lasso", build_lasso),
        ("MLP", build_mlp),
    ]:
        best = builder()
        results.append(
            fit_and_report(
                name,
                best,
                X_tr,
                X_te,
                y_tr,
                y_te,
                list(X.columns),
                means,
            )
        )

    print("\nSummary:")
    print(pd.DataFrame(results, columns=["Model", "MAE (delta)", "R²", "File"]))

    if quali_csv:
        print("\n--- Inference using best-MAE model ---")
        best_name = min(results, key=lambda r: r[1])[0].lower()
        predict_with_best('gbr', quali_csv)


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #
def _build_feature_frame(q, meta, drv_col, lap_col):
    feats = pd.DataFrame(columns=meta["feature_names"])

    if "start_grid" in feats.columns and "start_grid" in q.columns:
        feats["start_grid"] = pd.to_numeric(q["start_grid"], errors="coerce")

    if "best_quali_time" in feats.columns:
        if {"q1", "q2", "q3"}.issubset(q.columns):
            feats["best_quali_time"] = (
                q[["q1", "q2", "q3"]].apply(pd.to_numeric, errors="coerce").min(axis=1)
            )
        elif lap_col:
            feats["best_quali_time"] = pd.to_numeric(q[lap_col], errors="coerce")

    if "season_avg_pos_delta" in feats.columns:
        season_table = pd.read_csv(FEATURE_CSV)[
            ["driver", "year", "season_avg_pos_delta"]
        ]
        latest_year = season_table["year"].max()
        lookup = (
            season_table[season_table["year"] == latest_year]
            .groupby("driver", as_index=True)["season_avg_pos_delta"]
            .mean()
        )
        feats["season_avg_pos_delta"] = q[drv_col].map(lookup)

    # load full engineered_features.csv
    all_feats   = pd.read_csv(FEATURE_CSV)
    latest_year = all_feats["year"].max()

    # race_finish_pct & driver_strength (one row per driver)
    season = (
        all_feats[all_feats["year"] == latest_year]
        .loc[:, ["driver", "race_finish_pct", "driver_strength"]]
        .drop_duplicates(subset=["driver"])
        .set_index("driver")
    )
    feats["race_finish_pct"] = q[drv_col].map(season["race_finish_pct"])
    feats["driver_strength"] = q[drv_col].map(season["driver_strength"])

    # team_strength: drop on driver so index is unique
    team_lookup = (
        all_feats[all_feats["year"] == latest_year]
        .loc[:, ["driver", "team_strength"]]
        .drop_duplicates(subset=["driver"])
        .set_index("driver")["team_strength"]
    )
    feats["team_strength"] = q[drv_col].map(team_lookup)

    # ── fill any remaining NaNs with training‐set means ────────────────
    for col in feats.columns:
        feats[col] = feats[col].fillna(meta["feature_means"].get(col, 0.0))

    return feats


def predict_with_best(best_tag, quali_csv):
    reg = joblib.load(f"{best_tag}_model.pkl")
    meta = joblib.load(f"{best_tag}_meta.pkl")

    q = pd.read_csv(quali_csv)
    cols = {c.lower().strip(): c for c in q.columns}
    drv_col = next(
        (cols[k] for k in cols if k in {"driver", "driver_abbreviation", "abbr"}), None
    )
    lap_col = next(
        (
            cols[k]
            for k in cols
            if (
                k.startswith("qualifying")
                or "lap" in k
                or k in {"q1", "q2", "q3", "best_quali_time"}
            )
        ),
        None,
    )

    X_inf = _build_feature_frame(q, meta, drv_col, lap_col)
    q["PredDelta"] = reg.predict(X_inf)

    # Compute a continuous finishing “score”: lower is better
    q["PredScore"] = q["start_grid"] - q["PredDelta"]

    # Rank by that score
    q = q.sort_values("PredScore").reset_index(drop=True)
    q["PredFinalPosition"] = np.arange(1, len(q) + 1)

    print(
        q[[
            drv_col,
            "start_grid",
            "PredFinalPosition"
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    qualifying_csv_file: str = "qualifying_saudi.csv"
    main(qualifying_csv_file)
