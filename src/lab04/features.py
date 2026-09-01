"""Признаки на уровне игрока (минимум 10)."""

import pandas as pd


def player_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("player_id")
    f = pd.DataFrame({
        "n_sessions": g.session_id.nunique(),
        "avg_len": g.length_s.mean(),
        "total_len": g.length_s.sum(),
        "kills_per_session": g.kills.mean(),
        "deaths_per_session": g.deaths.mean(),
        "kd_ratio": (g.kills.sum() + 1) / (g.deaths.sum() + 1),
        "explored_mean": g.explored.mean(),
        "chats_per_session": g.chats.mean(),
        "active_days": g.day.nunique(),
        "last_day": g.day.max(),
    })
    f["sessions_per_active_day"] = f.n_sessions / f.active_days
    return f
