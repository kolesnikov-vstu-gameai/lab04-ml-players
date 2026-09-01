from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cluster(X, k_range=range(2, 8), seed=42):
    Xs = StandardScaler().fit_transform(X)
    scores = {k: silhouette_score(Xs, KMeans(k, n_init=10, random_state=seed).fit_predict(Xs)) for k in k_range}
    best = max(scores, key=scores.get)
    labels = KMeans(best, n_init=10, random_state=seed).fit_predict(Xs)
    return best, scores, labels


def churn_model(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    m = RandomForestClassifier(300, random_state=seed).fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    return m, auc
