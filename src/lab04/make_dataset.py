"""Синтетический датасет: 1000+ игроков, 30 дней, 3 архетипа (aggressive / explorer / social)."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "events.csv"
ARCHETYPES = {
    "aggressive": dict(kills=8, deaths=4, explored=0.3, chats=1, sessions=12),
    "explorer": dict(kills=2, deaths=2, explored=0.9, chats=2, sessions=9),
    "social": dict(kills=3, deaths=3, explored=0.5, chats=10, sessions=15),
}


def make(n_players: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        arch = rng.choice(list(ARCHETYPES))
        p = ARCHETYPES[arch]
        churned = rng.random() < (0.35 if arch == "aggressive" else 0.2)
        n_sess = max(1, int(rng.poisson(p["sessions"]) * (0.4 if churned else 1)))
        for s in range(n_sess):
            rows.append(dict(player_id=f"p{pid}", session_id=f"p{pid}_s{s}", day=int(rng.integers(0, 30)),
                             kills=rng.poisson(p["kills"]), deaths=rng.poisson(p["deaths"]),
                             explored=float(np.clip(rng.normal(p["explored"], 0.1), 0, 1)),
                             chats=rng.poisson(p["chats"]), length_s=float(rng.gamma(3, 200)),
                             archetype=arch, churned=int(churned)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    df = make()
    df.to_csv(OUT, index=False)
    print(f"{len(df)} событий-сессий, {df.player_id.nunique()} игроков → {OUT}")
