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
FEATURE_CSV  = "engineered_features.csv"
TARGET_COL   = "race_avg_lap"
ID_COLS      = ["driver", "team", "year", "round", "final_position", "winner"]
RANDOM_SEED  = 42
CV_FOLDS     = 5
N_ITER_GBR   = 50      # # random combos for GBR
N_ITER_MLP   = 40      # # random combos for MLP
###############################################################################


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _prep_dataframe(df: pd.DataFrame):
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    if TARGET_COL in X.columns:
        X = X.drop(columns=TARGET_COL)
    return X.fillna(X.mean(numeric_only=True))


def load_training_data(path=FEATURE_CSV):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise KeyError(f"{TARGET_COL} missing – run feature_engineering.py first.")
    return _prep_dataframe(df), df[TARGET_COL]


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def build_gbr():
    base = GradientBoostingRegressor(random_state=RANDOM_SEED)
    param_dist = {
        "n_estimators":   np.arange(100, 2600, 100),
        "learning_rate":  np.linspace(0.01, 0.15, 50),
        "max_depth":      np.arange(3, 11),
        "min_samples_split": np.arange(2, 16),
        "min_samples_leaf":  np.arange(1, 11),
        "subsample":      np.linspace(0.6, 2.0, 50),
        "max_features":   ["sqrt", "log2", None],
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
    # Standardise first, then Lasso (α grid)
    pipe = Pipeline(
        steps=[
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
        steps=[
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
    print(f"{name:<9} | MAE = {mae:6.3f} s | R² = {r2:6.3f}")

    out_path = f"{name.lower()}_model.pkl"
    meta_out = f"{name.lower()}_meta.pkl"
    joblib.dump(model, out_path)
    joblib.dump({"feature_names": feature_names, "feature_means": means}, meta_out)
    return name, mae, r2, out_path


def main(args):
    X, y = load_training_data()
    means = X.mean(numeric_only=True).to_dict()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Model      |  Test‑set metrics")
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

    # nice comparison table
    print("\nSummary:")
    print(pd.DataFrame(results, columns=["Model", "MAE (s)", "R²", "File"]))

    if args.quali_csv:
        print("\n--- Inference using best‑MAE model ---")
        best_name = min(results, key=lambda r: r[1])[0].lower()
        predict_with_best(best_name, args.quali_csv)


# --------------------------------------------------------------------------- #
# Inference helper (re‑uses earlier logic)
# --------------------------------------------------------------------------- #
def _build_feature_frame(q, meta, drv_col, lap_col):
    feats = pd.DataFrame(columns=meta["feature_names"])

    # quali time supplied by the user
    if "best_quali_time" in feats.columns:
        feats["best_quali_time"] = pd.to_numeric(q[lap_col], errors="coerce")

    # season‑average pos delta from engineered CSV if available
    if "season_avg_pos_delta" in feats.columns:
        season_table = pd.read_csv(FEATURE_CSV)[["driver", "year", "season_avg_pos_delta"]]
        latest_year   = season_table["year"].max()          # 2025 in your use‑case
        lookup = (
            season_table[season_table["year"] == latest_year]
            .set_index("driver")["season_avg_pos_delta"]
        )
        feats["season_avg_pos_delta"] = q[drv_col].map(lookup)

    # everything else → mean imputation
    for col in feats.columns:
        feats[col] = feats[col].fillna(meta["feature_means"].get(col, 0.0))
    return feats


def predict_with_best(best_tag, quali_csv):
    reg  = joblib.load(f"{best_tag}_model.pkl")
    meta = joblib.load(f"{best_tag}_meta.pkl")

    q = pd.read_csv(quali_csv)
    cols    = {c.lower().strip(): c for c in q.columns}
    drv_col = next((cols[k] for k in cols if k in {"driver", "driver_abbreviation", "abbr"}), None)
    lap_col = next((cols[k] for k in cols if k.startswith("qualifying") or "lap" in k), None)

    X_inf = _build_feature_frame(q, meta, drv_col, lap_col)
    q["PredRaceTime (s)"] = reg.predict(X_inf)
    print(q[[drv_col, "PredRaceTime (s)"]].sort_values("PredRaceTime (s)"))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare GBR, Lasso & MLPRegressor on lap‑time prediction."
    )
    parser.add_argument(
        "--quali_csv",
        type=str,
        help="Minimal quali CSV → predict with best model",
        default=None,
    )
    main(parser.parse_args())
