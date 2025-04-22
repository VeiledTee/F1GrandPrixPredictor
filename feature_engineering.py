from __future__ import annotations
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CACHE_DIR = "cache"  # FastF1 disk cache
MISSING_THRESHOLD = 0.70  # drop columns when NaN fraction exceeds this


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _time_to_seconds(val) -> float:
    """Convert Timedelta / 'm:ss.xxx' / float to seconds (NaN on failure)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    if isinstance(val, (pd.Timedelta, np.timedelta64)):
        return float(pd.to_timedelta(val).total_seconds())
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return np.nan
        if ":" in val:
            mins, secs = val.split(":")
            try:
                return int(mins) * 60 + float(secs)
            except ValueError:
                return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan


def _standardise_quali_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a minimal quali CSV → driver, best_quali_time, start_grid."""
    cols = {c.lower().strip(): c for c in df.columns}
    drv = next(
        (cols[k] for k in cols if k in {"driver", "driver_abbreviation", "abbr"}), None
    )
    q_parts: list[str] = [c for c in cols if c in {"q1", "q2", "q3"}]
    if q_parts:
        lap_series: pd.Series = (
            df[[cols[c] for c in q_parts]].applymap(_time_to_seconds).min(axis=1)
        )
    else:
        lap = next(
            (
                cols[k]
                for k in cols
                if k
                in {
                    "fastest lap",
                    "fastest_lap",
                    "fastest_lap_time",
                    "lap_time",
                    "time",
                }
            ),
            None,
        )
        if lap is None:
            raise ValueError(
                "Quali CSV must contain a lap‑time column or Q1/Q2/Q3 columns"
            )
        lap_series = df[lap].apply(_time_to_seconds)

    grd = next(
        (
            cols[k]
            for k in cols
            if k in {"position", "grid_position", "start_grid", "pos", "grid"}
        ),
        None,
    )
    if drv is None:
        raise ValueError("Quali CSV must contain at least driver and lap‑time columns")
    out = pd.DataFrame()
    out["driver"] = df[drv]
    out["best_quali_time"] = lap_series
    if grd is not None:
        out["start_grid"] = pd.to_numeric(df[grd], errors="coerce")
    return out


def _fetch_quali_summary(year: int, rnd: int) -> pd.DataFrame:
    """Download qualifying fastest laps via FastF1 (seconds)."""
    import fastf1

    fastf1.Cache.enable_cache(CACHE_DIR)
    try:
        session = fastf1.get_session(year, rnd, "Q")
        session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps
        summary = (
            laps.groupby("Driver")["LapTime"]
            .min()
            .reset_index()
            .rename(columns={"Driver": "driver"})
        )
        summary["best_quali_time"] = summary["LapTime"].dt.total_seconds()
        summary = summary.drop(columns="LapTime")
        summary["year"], summary["round"] = year, rnd
        return summary[["year", "round", "driver", "best_quali_time"]]
    except Exception as e:
        print(f"⚠️  Could not fetch quali for {year}‑R{rnd}: {e}")
        return pd.DataFrame(columns=["year", "round", "driver", "best_quali_time"])


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def engineer_features(
    df: pd.DataFrame,
    *,
    external_quali_df: Optional[pd.DataFrame] = None,
    season_year: Optional[int] = None,
    gp_round: Optional[int] = None,
) -> pd.DataFrame:
    """Return one feature row per **driver‑race**.

    Parameters
    ----------
    df : pd.DataFrame
        Raw laps dataframe (practice, quali, race – but race only is fine).
    external_quali_df : pd.DataFrame, optional
        Minimal CSV for the current GP (driver, lap time, grid pos).
    season_year, gp_round : int, optional
        Used to label *external_quali_df* when `df` spans multiple races.
    """
    df = df.copy()

    # ------------------------------------------------------------------- #
    # 0. Coerce lap_time to seconds everywhere it appears
    # ------------------------------------------------------------------- #
    if "lap_time" in df.columns:
        df["lap_time"] = df["lap_time"].apply(_time_to_seconds)

    # ------------------------------------------------------------------- #
    # 1. Qualifying summary
    # ------------------------------------------------------------------- #
    quali_df = df[df["session_type"] == "Q"].copy()

    # (a) merge user‑supplied mini‑Q
    if external_quali_df is not None and not external_quali_df.empty:
        ext = _standardise_quali_columns(external_quali_df)
        if season_year is None:
            y_unique = df["year"].unique()
            if len(y_unique) == 1:
                season_year = int(y_unique[0])
            else:
                raise ValueError(
                    "season_year must be provided (multiple seasons detected)."
                )
        if gp_round is None:
            r_unique = df["round"].unique()
            if len(r_unique) == 1:
                gp_round = int(r_unique[0])
            else:
                raise ValueError(
                    "gp_round must be provided (multiple rounds detected)."
                )
        ext["year"], ext["round"] = season_year, gp_round
        quali_df = pd.concat(
            [quali_df, ext.assign(session_type="Q")], ignore_index=True
        )

    # (b) auto‑fetch when no quali present
    if quali_df.empty or quali_df["lap_time"].isna().all():
        fetched: list[pd.DataFrame] = []
        for yr, rnd in (
            df.loc[df["session_type"] == "R", ["year", "round"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            fetched.append(_fetch_quali_summary(int(yr), int(rnd)))
        quali_df = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()

    # summarise
    if "best_quali_time" in quali_df.columns:
        keep = ["year", "round", "driver", "best_quali_time"]
        if "start_grid" in quali_df.columns:
            keep.append("start_grid")
        quali_summary = quali_df[keep].copy()
    else:  # raw quali laps
        quali_df["lap_time"] = quali_df["lap_time"].apply(_time_to_seconds)
        quali_summary = (
            quali_df.groupby(["year", "round", "driver"])
            .agg(best_quali_time=("lap_time", "min"))
            .reset_index()
        )

    # ------------------------------------------------------------------- #
    # 2. Practice summary
    # ------------------------------------------------------------------- #
    practice_df = df[df["session_type"].isin(["FP1", "FP2", "FP3"])]
    if not practice_df.empty:
        practice_df["lap_time"] = practice_df["lap_time"].apply(_time_to_seconds)
        practice_summary = (
            practice_df.groupby(["year", "round", "driver"])
            .agg(
                practice_avg_lap=("lap_time", "mean"),
                practice_std_lap=("lap_time", "std"),
                practice_best_lap=("lap_time", "min"),
            )
            .reset_index()
        )
    else:
        practice_summary = pd.DataFrame()

    # ------------------------------------------------------------------- #
    # 3. Race‑level aggregations
    # ------------------------------------------------------------------- #
    race_df = df[df["session_type"] == "R"].copy()
    total_laps_gp = race_df.groupby(["year", "round"])["lap_number"].transform("max")
    race_df["finish_flag"] = (race_df["lap_number"] == total_laps_gp).astype(int)

    race_df["lap_time"] = race_df["lap_time"].apply(_time_to_seconds)
    race_df.sort_values(["year", "round", "driver", "lap_number"], inplace=True)

    grid_start = (
        race_df.groupby(["year", "round", "driver"])
        .agg(start_grid=("position", "first"))
        .reset_index()
    )

    race_df["compound_change"] = (
        race_df.groupby(["year", "round", "driver"])["compound"]
        .transform(lambda x: x != x.shift(1))
        .fillna(False)
    )
    race_df["num_stops"] = race_df.groupby(["year", "round", "driver"])[
        "compound_change"
    ].transform("sum")
    lap1 = (
        race_df[race_df["lap_number"] == 1]
        .loc[:, ["year", "round", "driver", "position"]]
        .rename(columns={"position": "lap1_position"})
    )
    race_df["is_caution"] = race_df["track_status"].apply(lambda x: 0 if x == 1 else 1)
    race_df["rain_lap"] = race_df.get("weather_rainfall", False).astype(int)

    # tyre usage fractions
    compound_usage = race_df.pivot_table(
        index=["year", "round", "driver"],
        columns="compound",
        values="lap_number",
        aggfunc="count",
    ).fillna(0)
    compound_usage_frac = compound_usage.div(compound_usage.sum(axis=1), axis=0)
    compound_usage_frac.columns = [
        f"pct_{c.lower()}" for c in compound_usage_frac.columns
    ]
    compound_usage_frac = compound_usage_frac.reset_index()

    race_features = (
        race_df.groupby(["year", "round", "driver", "team"])
        .agg(
            race_avg_lap=("lap_time", "mean"),
            race_std_lap=("lap_time", "std"),
            num_stops=("num_stops", "max"),
            final_position=("position", "last"),
            finish_flag=("finish_flag", "max"),
            pct_caution_laps=("is_caution", "mean"),
            pct_rain_laps=("rain_lap", "mean"),
            total_laps=("lap_number", "max"),
        )
        .reset_index()
    )

    race_features["avg_stint_len"] = race_features["total_laps"] / (
        race_features["num_stops"] + 1
    )
    race_features.drop(columns="total_laps", inplace=True)

    # ------------------------------------------------------------------- #
    # 4. Merge all blocks
    # ------------------------------------------------------------------- #
    features = race_features.merge(
        quali_summary, on=["year", "round", "driver"], how="left"
    )
    if not practice_summary.empty:
        features = features.merge(
            practice_summary, on=["year", "round", "driver"], how="left"
        )

    features = (
        features.merge(
            grid_start,
            on=["year", "round", "driver"],
            how="left",
            suffixes=("", "_race"),
        )
        .merge(compound_usage_frac, on=["year", "round", "driver"], how="left")
        .merge(lap1, on=["year", "round", "driver"], how="left")
    )
    if "start_grid_x" in features.columns:
        features["start_grid"] = features["start_grid_x"].combine_first(
            features["start_grid_race"]
        )
        features.drop(columns=["start_grid_x", "start_grid_race"], inplace=True)

    # ------------------------------------------------------------------- #
    # 5. Targets & clean‑up
    # ------------------------------------------------------------------- #
    features["winner"] = (features["final_position"] == 1).astype(int)
    features["pos_change"] = features["start_grid"] - features["final_position"]
    features["lap1_pos_change"] = features["start_grid"] - features["lap1_position"]

    season_grp = features.groupby(["year", "driver"])
    features["season_avg_pos_delta"] = season_grp["pos_change"].transform("mean")
    # driver strength = mean finishing position  (lower = better)
    features["driver_strength"] = season_grp["final_position"].transform("mean")
    # driver’s finish rate
    fin_pct = season_grp["finish_flag"].transform("mean")
    features["race_finish_pct"] = fin_pct
    # team strength (same season)
    team_strength = features.groupby(["year", "team"])["final_position"].transform(
        "mean"
    )
    features["team_strength"] = team_strength

    # drop sparse columns
    sparse_cols: Sequence[str] = [
        c
        for c in features.columns
        if features[c].isna().mean() > MISSING_THRESHOLD and c != "best_quali_time"
    ]
    return features.drop(columns=sparse_cols)


def main():
    raw = pd.read_csv("race_data_full.csv")
    mini_q = pd.read_csv("qualifying_saudi.csv")
    feats = engineer_features(
        raw, external_quali_df=mini_q, season_year=2025, gp_round=5
    )
    feats.to_csv("engineered_features.csv", index=False)
    print("✅ engineered_features.csv saved →", feats.shape)


if __name__ == "__main__":
    main()
