import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CURRENT_RACE = "Saudi Arabia"


def load_data():
    df = pd.read_csv("engineered_features.csv")

    # Use only features available at prediction time
    df = df[["driver", "best_quali_time", "race_avg_lap"]].dropna()

    return df


def train_models(X_train, y_train):
    models = {
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def predict_winner(models, quali_df):
    predictions = {}
    for name, model in models.items():
        X_input = quali_df[["quali_fastest_lap"]].copy()
        X_input.columns = ["best_quali_time"]
        predicted_laps = model.predict(X_input)
        quali_df["predicted_lap"] = predicted_laps
        winner_row = quali_df.loc[quali_df["predicted_lap"].idxmin()]

        predictions[name] = {
            "driver": winner_row["driver"],
            "predicted_lap": winner_row["predicted_lap"],
        }

    return predictions


def evaluate_models(models, X_test, y_test):
    errors = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        errors[name] = mae
    return errors


def main():
    # --- Load historical data ---
    df = load_data()
    X = df[["best_quali_time"]]
    y = df["race_avg_lap"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train models ---
    models = train_models(X_train, y_train)

    # --- Load 2025 Saudi Arabia GP Quali times ---
    quali_df = pd.read_csv(
        "2025_saudi_quali.csv"
    )  # must have: driver,quali_fastest_lap
    quali_df = quali_df.rename(columns={"quali_fastest_lap": "best_quali_time"})

    # --- Predict winners ---
    predictions = predict_winner(models, quali_df)
    errors = evaluate_models(models, X_test, y_test)

    # --- Display Results ---
    for name in models.keys():
        print(f"\n🏁 Predicted 2025 Saudi Arabia GP Winner ({name}) 🏁")
        print(f"Driver: {predictions[name]['driver']}")
        print(f"Predicted Race Time: {predictions[name]['predicted_lap']:.2f}s")
        print(f"🔍 Model Error (MAE): {errors[name]:.2f} seconds")


if __name__ == "__main__":
    main()
