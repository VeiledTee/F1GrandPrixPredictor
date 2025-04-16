# data_collector.py
import fastf1
import pandas as pd
from pathlib import Path

# Enable cache to speed up future loads
fastf1.Cache.enable_cache(Path("cache"))

# Define session types you want to extract
SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'R']  # Practice, Quali, Sprint, Race


def collect_session_data(year, rnd, session_type):
    try:
        session = fastf1.get_session(year, rnd, session_type)
        session.load()
        laps = session.laps
        weather = session.weather_data

        data = []

        for lap in laps.iterlaps():
            data.append({
                'year': year,
                'round': rnd,
                'session_type': session_type,
                'driver': lap.Driver,
                'team': lap.Team,
                'lap_number': lap.LapNumber,
                'lap_time': lap.LapTime.total_seconds() if lap.LapTime else None,
                'position': lap.Position,
                'compound': lap.Compound,
                'track_status': lap.TrackStatus,
                'start_time': lap.LapStartTime,
                'quali_pos': session.results.loc[lap.Driver]['GridPosition']
                    if lap.Driver in session.results.index and 'GridPosition' in session.results.columns else None,
                'weather_air_temp': weather.loc[lap.LapStartTime]['AirTemp'] if weather is not None and lap.LapStartTime in weather.index else None,
                'weather_track_temp': weather.loc[lap.LapStartTime]['TrackTemp'] if weather is not None and lap.LapStartTime in weather.index else None,
                'weather_humidity': weather.loc[lap.LapStartTime]['Humidity'] if weather is not None and lap.LapStartTime in weather.index else None,
                'weather_rainfall': weather.loc[lap.LapStartTime]['Rainfall'] if weather is not None and lap.LapStartTime in weather.index else None,
                'weather_wind_speed': weather.loc[lap.LapStartTime]['WindSpeed'] if weather is not None and lap.LapStartTime in weather.index else None
            })
        return data
    except Exception as e:
        print(f"Failed to load {session_type} session for {year} Round {rnd}: {e}")
        return []


def collect_all_sessions(start_year=2018, end_year=2024):
    all_data = []

    for year in range(start_year, end_year + 1):
        for rnd in range(1, 24):  # Increase range if new season adds more races
            for session_type in SESSION_TYPES:
                session_data = collect_session_data(year, rnd, session_type)
                all_data.extend(session_data)

    return pd.DataFrame(all_data)


def main():
    df = collect_all_sessions()
    df.to_csv("raw_f1_data_full.csv", index=False)
    print("Saved full session data to raw_f1_data_full.csv")


if __name__ == '__main__':
    main()
