"""From logs to an evaluation set.

1. Near-duplicates (cosine similarity at least 0.97) collapse into one candidate.
2. Clusters: HDBSCAN over (PCA-reduced) sentence embeddings finds the kinds of request; each is
   named by its most distinctive words. Points in no cluster are outliers: novel requests.
3. Sampling a labelling budget, three ways, each with inclusion probabilities so the
   labelled set still estimates quality over all traffic without bias:
   - uniform: every candidate equally likely;
   - stratified: each cluster (and the outliers) gets its share of the budget;
   - boosted: oversample what the signals flag (low model confidence, outliers, negative
     feedback), weighted back by inverse inclusion probability (Horvitz-Thompson, Hajek form).
4. The chosen interactions become test cases in a versioned JSON-lines file, the version
   being a hash of the content.
"""
import hashlib
import json

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def near_duplicates(vectors, threshold=0.97):
    """Group index for each row: rows within `threshold` cosine of an earlier row join its group."""
    group = np.arange(len(vectors))
    for i in range(len(vectors)):
        if group[i] == i:
            sims = vectors[i + 1:] @ vectors[i]
            for j in np.flatnonzero(sims >= threshold) + i + 1:
                if group[j] == j:
                    group[j] = i
    return group


def clusters(vectors, texts, min_size=5, dims=32, words=3, seed=0):
    """(label per row, -1 for outliers; {label: name}). Density clustering struggles in 384
    dimensions (it found 10 clusters on Banking77); on a 32-dimension PCA projection it finds
    the request types (71, mostly one intent each)."""
    reduced = PCA(dims, random_state=seed).fit_transform(vectors) if vectors.shape[1] > dims else vectors
    labels = HDBSCAN(min_cluster_size=min_size, copy=True).fit_predict(reduced)
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    x = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())
    names = {}
    for c in sorted(set(labels) - {-1}):
        score = np.asarray(x[labels == c].mean(0)).ravel() - np.asarray(x[labels != c].mean(0)).ravel()
        names[int(c)] = " / ".join(vocab[np.argsort(-score)[:words]])
    return labels, names


def signal(confidence, outlier, feedback=None, boost=4.0):
    """How much to oversample each interaction: 1, plus `boost` times its trouble signals."""
    trouble = (1 - np.asarray(confidence)) + 0.5 * np.asarray(outlier, float)
    if feedback is not None:
        trouble = trouble + np.isin(feedback, ["thumbs_down", "retried", "edited"])
    return 1 + boost * trouble


def inclusion(weights, n):
    """Inclusion probabilities proportional to weight, summing to n (capped at 1)."""
    w = np.asarray(weights, float)
    pi = np.zeros_like(w)
    fixed = np.zeros(len(w), bool)
    for _ in range(len(w)):
        free = n - fixed.sum()
        pi[~fixed] = free * w[~fixed] / w[~fixed].sum()
        over = (pi > 1) & ~fixed
        if not over.any():
            break
        fixed |= over
        pi[fixed] = 1
    return np.minimum(pi, 1)


def sample(strategy, n, rng, strata=None, weights=None):
    """(chosen indices, their inclusion probabilities). Poisson sampling: each row in independently."""
    size = len(strata if strata is not None else weights)
    if strategy == "uniform":
        pi = np.full(size, n / size)
    elif strategy == "stratified":
        pi = np.zeros(size)
        for s in np.unique(strata):
            members = strata == s
            pi[members] = max(n * members.mean(), 1) / members.sum()     # every stratum gets at least one expected pick
        pi = np.minimum(pi * n / pi.sum(), 1)
    else:
        pi = inclusion(weights, n)
    chosen = np.flatnonzero(rng.random(size) < pi)
    return chosen, pi[chosen]


def estimate(values, pi):
    """Hajek estimate of the population mean from a probability sample."""
    w = 1 / np.asarray(pi)
    return float((w * np.asarray(values)).sum() / w.sum())


def write(cases, path, meta):
    """Test cases to JSON lines, with a manifest; the version is a hash of the cases."""
    body = "\n".join(json.dumps(c, sort_keys=True) for c in cases) + "\n"
    version = hashlib.sha256(body.encode()).hexdigest()[:12]
    path.write_text(body)
    path.with_suffix(".manifest.json").write_text(json.dumps({"version": version, "cases": len(cases), **meta}, indent=1))
    return version
