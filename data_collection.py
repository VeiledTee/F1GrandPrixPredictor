import fastf1
import pandas as pd
from pathlib import Path

# Create cache directory if it doesn't exist
cache_path = Path("cache")
cache_path.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(cache_path)

# Define session types you want to extract
# SESSION_TYPES = ["Q", "S", "R"]  # Quali, Sprint, Race
SESSION_TYPES = ["R"]  # Quali, Sprint, Race


def collect_session_data(year, rnd, session_type):
    try:
        session = fastf1.get_session(year, rnd, session_type)
        session.load()
        laps = session.laps.pick_accurate()  # use only accurate laps
        weather = session.weather_data
        event_name = session.event["EventName"]

        data = []

        for _, lap in laps.iterlaps():
            driver_abbr = lap["Driver"]
            lap_start_time = lap["LapStartTime"]

            weather_point = None
            try:
                if weather is not None:
                    # Match based on 'Time' column in weather dataframe
                    weather_point = weather.iloc[
                        (weather["Time"] - lap_start_time).abs().argsort()[:1]
                    ].squeeze()
            except Exception:
                pass

            data.append(
                {
                    "year": year,
                    "round": rnd,
                    "session_type": session_type,
                    "event_name": event_name,
                    "driver": driver_abbr,
                    "team": lap["Team"],
                    "lap_number": lap["LapNumber"],
                    "lap_time": (
                        lap["LapTime"].total_seconds()
                        if pd.notnull(lap["LapTime"])
                        else None
                    ),
                    "grid_position": session.results["Position"],
                    "position": lap["Position"],
                    "compound": lap["Compound"],
                    "track_status": lap["TrackStatus"],
                    "start_time": lap_start_time,
                    "weather_air_temp": (
                        weather_point["AirTemp"] if weather_point is not None else None
                    ),
                    "weather_track_temp": (
                        weather_point["TrackTemp"]
                        if weather_point is not None
                        else None
                    ),
                    "weather_humidity": (
                        weather_point["Humidity"] if weather_point is not None else None
                    ),
                    "weather_rainfall": (
                        weather_point["Rainfall"] if weather_point is not None else None
                    ),
                    "weather_wind_speed": (
                        weather_point["WindSpeed"]
                        if weather_point is not None
                        else None
                    ),
                }
            )
        print(f"✅ Loaded {session_type} session for {year} Round {rnd} ({event_name})")
        return data
    except Exception as e:
        print(f"❌ Failed to load {session_type} session for {year} Round {rnd}: {e}")
        return []


def collect_all_sessions(start_year=2024, end_year=2025):
    all_data = []
    for year in range(start_year, end_year + 1):
        for rnd in range(1, 24):  # Update based on max number of rounds
            for session_type in SESSION_TYPES:
                session_data = collect_session_data(year, rnd, session_type)
                all_data.extend(session_data)
    return pd.DataFrame(all_data)


def main():
    df = collect_all_sessions()
    df.to_csv("race_data_full.csv", index=False)
    print("📦 Saved full session data to raw_f1_data_full.csv")


if __name__ == "__main__":
    main()
