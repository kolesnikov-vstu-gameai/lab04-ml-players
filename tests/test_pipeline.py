from lab04.features import player_features
from lab04.make_dataset import make
from lab04.models import churn_model, cluster


def test_end_to_end_small():
    df = make(n_players=150, seed=1)
    f = player_features(df)
    assert f.shape[1] >= 10
    k, _, labels = cluster(f)
    assert 2 <= k <= 7 and len(labels) == len(f)
    y = df.groupby("player_id").churned.max().loc[f.index]
    _, auc = churn_model(f, y)
    assert auc > 0.5
