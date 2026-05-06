"""Lightweight LSA experiment — two phases:
Phase 1: Quick screening on 800-row train sample with fewer configs
Phase 2: Validate top-3 on full training set
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.global_config import SENTIMENT_LABELS, SentimentLabel, TRAIN_ASPECTS

SEED = 25
TRAIN_CSV = "statics/datasets/training.csv"
VAL_CSV = "statics/datasets/validate.csv"
SENT_3 = [l for l in SENTIMENT_LABELS if l != "notmentioned"]
NM_IDX = SENTIMENT_LABELS.index(SentimentLabel.NOTMENTIONED.value)
SENT_IDXS = [SENTIMENT_LABELS.index(s) for s in SENT_3]

warnings.filterwarnings("ignore")


def load_data(path, max_rows=None):
    df = pd.read_csv(path)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=SEED).reset_index(drop=True)
    for a in TRAIN_ASPECTS:
        df[a] = df[a].fillna(SentimentLabel.NOTMENTIONED.value)
    texts = df["text"].fillna("").astype(str).to_numpy()
    lmap = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
    y = {a: df[a].map(lmap).fillna(NM_IDX).astype(int).values for a in TRAIN_ASPECTS}
    return texts, y


def evaluate(y_true, y_pred):
    at = np.concatenate([y_true[a] for a in TRAIN_ASPECTS])
    ap = np.concatenate([y_pred[a] for a in TRAIN_ASPECTS])
    det_t = (at != NM_IDX).astype(int)
    det_p = (ap != NM_IDX).astype(int)
    det_f1 = f1_score(det_t, det_p, average="binary", zero_division=0)
    mask = at != NM_IDX
    sent_f1 = f1_score(at[mask], ap[mask], labels=SENT_IDXS, average="macro", zero_division=0) if mask.sum() else 0.0
    f1_4 = f1_score(at, ap, labels=list(range(len(SENTIMENT_LABELS))), average="macro", zero_division=0)
    coverage = (ap != NM_IDX).mean()
    return {"det_f1": det_f1, "sent_f1": sent_f1, "f1_4class": f1_4, "coverage": coverage}


CONFIGS = [
    # (name, tfidf_kw, n_components, clf_factory)
    ("LR_C1_10k_100",    dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
    ("LR_C1_10k_200",    dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  200,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
    ("LR_C10_10k_100",   dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=10.0, random_state=SEED)),
    ("LR_C01_10k_100",   dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=SEED)),
    ("LR_C1_20k_200",    dict(max_features=20000, ngram_range=(1,3), sublinear_tf=True, min_df=2),  200,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
    ("LR_C1_20k_300",    dict(max_features=20000, ngram_range=(1,3), sublinear_tf=True, min_df=2),  300,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
    ("SGD_log_10k_100",  dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, random_state=SEED)),
    ("SGD_log_20k_200",  dict(max_features=20000, ngram_range=(1,3), sublinear_tf=True, min_df=2),  200,
     lambda: SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, random_state=SEED)),
    ("SVC_10k_100",      dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: Pipeline([("sc", StandardScaler()), ("svc", LinearSVC(class_weight="balanced", max_iter=3000, random_state=SEED))])),
    ("SVC_20k_200",      dict(max_features=20000, ngram_range=(1,3), sublinear_tf=True, min_df=2),  200,
     lambda: Pipeline([("sc", StandardScaler()), ("svc", LinearSVC(class_weight="balanced", max_iter=3000, random_state=SEED))])),
    ("RF_100_10k_100",   dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=20, random_state=SEED, n_jobs=2)),
    ("RF_200_20k_200",   dict(max_features=20000, ngram_range=(1,3), sublinear_tf=True, min_df=2),  200,
     lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=25, random_state=SEED, n_jobs=2)),
    ("GB_100_10k_100",   dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2),  100,
     lambda: GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED)),
    # char ngrams
    ("LR_C1_char_15k_100", dict(max_features=15000, ngram_range=(2,5), analyzer="char_wb", sublinear_tf=True, min_df=2), 100,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
    ("LR_C1_15k_12_md90",  dict(max_features=15000, ngram_range=(1,2), sublinear_tf=True, min_df=3, max_df=0.90), 100,
     lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
]


def phase1_screen(train_texts, train_y, val_texts, val_y):
    """Quick screen all configs on a small sample."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: Screening {len(CONFIGS)} configs (train={len(train_texts)} rows)")
    print(f"{'='*70}")
    results = []
    for i, (name, tfidf_kw, n_comp, clf_fn) in enumerate(CONFIGS):
        t0 = time.time()
        tfidf = TfidfVectorizer(stop_words="english", **tfidf_kw)
        X_tr = tfidf.fit_transform(train_texts)
        X_val = tfidf.transform(val_texts)
        if n_comp >= X_tr.shape[1]:
            print(f"  [{i+1}/{len(CONFIGS)}] {name} -- skipped (n_comp >= features)")
            continue
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        X_tr_lsa = svd.fit_transform(X_tr)
        X_val_lsa = svd.transform(X_val)
        var_exp = svd.explained_variance_ratio_.sum()

        y_pred = {}
        for a in TRAIN_ASPECTS:
            clf = clf_fn()
            clf.fit(X_tr_lsa, train_y[a])
            y_pred[a] = clf.predict(X_val_lsa)

        m = evaluate(val_y, y_pred)
        elapsed = time.time() - t0
        results.append({"name": name, "var_exp": round(var_exp, 4), **{k: round(v, 4) for k, v in m.items()}, "time": round(elapsed, 1)})
        print(f"  [{i+1}/{len(CONFIGS)}] {name:30s}  det={m['det_f1']:.3f}  sent={m['sent_f1']:.3f}  4c={m['f1_4class']:.3f}  cov={m['coverage']:.3f}  ({elapsed:.1f}s)")

    return pd.DataFrame(results).sort_values("f1_4class", ascending=False)


def phase2_fulldata(top_names, val_texts, val_y):
    """Re-run top configs with full training data."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: Full training set for top {len(top_names)} configs")
    print(f"{'='*70}")
    full_texts, full_y = load_data(TRAIN_CSV, max_rows=None)
    print(f"  Full train: {len(full_texts)} rows")

    config_map = {name: (tfidf_kw, n_comp, clf_fn) for name, tfidf_kw, n_comp, clf_fn in CONFIGS}
    results = []
    for name in top_names:
        tfidf_kw, n_comp, clf_fn = config_map[name]
        t0 = time.time()
        tfidf = TfidfVectorizer(stop_words="english", **tfidf_kw)
        X_tr = tfidf.fit_transform(full_texts)
        X_val = tfidf.transform(val_texts)
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        X_tr_lsa = svd.fit_transform(X_tr)
        X_val_lsa = svd.transform(X_val)

        y_pred = {}
        for a in TRAIN_ASPECTS:
            clf = clf_fn()
            clf.fit(X_tr_lsa, full_y[a])
            y_pred[a] = clf.predict(X_val_lsa)

        m = evaluate(val_y, y_pred)
        elapsed = time.time() - t0
        results.append({"name": name, **{k: round(v, 4) for k, v in m.items()}, "time": round(elapsed, 1)})
        print(f"  {name:30s}  det={m['det_f1']:.3f}  sent={m['sent_f1']:.3f}  4c={m['f1_4class']:.3f}  cov={m['coverage']:.3f}  ({elapsed:.1f}s)")

    return pd.DataFrame(results).sort_values("f1_4class", ascending=False)


def main():
    print("Loading data...")
    train_small, train_y_small = load_data(TRAIN_CSV, max_rows=800)
    val_texts, val_y = load_data(VAL_CSV, max_rows=None)
    print(f"  Phase 1 sample: {len(train_small)} train, {len(val_texts)} val")

    p1 = phase1_screen(train_small, train_y_small, val_texts, val_y)
    print(f"\n--- Phase 1 Leaderboard ---")
    print(p1.to_string(index=False))

    top3 = p1.head(3)["name"].tolist()
    print(f"\nTop 3 for Phase 2: {top3}")

    p2 = phase2_fulldata(top3, val_texts, val_y)
    print(f"\n--- Phase 2 Results (full data) ---")
    print(p2.to_string(index=False))

    best = p2.iloc[0]
    print(f"\n{'='*70}")
    print(f"WINNER: {best['name']}")
    print(f"  Detection F1:   {best['det_f1']:.4f}")
    print(f"  Sentiment F1:   {best['sent_f1']:.4f}")
    print(f"  4-class F1:     {best['f1_4class']:.4f}")
    print(f"  Coverage:       {best['coverage']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
