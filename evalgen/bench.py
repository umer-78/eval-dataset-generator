"""The generator on Banking77: 3,080 logged support messages, answered by a production model
(logistic regression on sentence embeddings, trained on the other 10,003 messages). A
labelling budget of 200 cases is spent three ways, 300 times each. Measured: how many of the
model's failures the set captures, how much of the traffic's variety it covers (intents; the
builder never sees them), and how far its accuracy estimate lands from the model's true
accuracy on all the logs."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data
from .build import clusters, estimate, near_duplicates, sample, signal, write
from .embed import Embedder

RESULTS = Path(__file__).resolve().parent.parent / "results"


def logs():
    train, test = data.messages("train"), data.messages("test")
    emb = Embedder()
    xtr, xte = emb.cached([t for t, _ in train]), emb.cached([t for t, _ in test])
    model = LogisticRegression(C=10, max_iter=3000).fit(xtr, [c for _, c in train])
    proba = model.predict_proba(xte)
    return test, xte, model.classes_[proba.argmax(1)], proba.max(1)


def run(budget=200, reps=300, seed=0):
    test, x, pred, conf = logs()
    gold = np.array([c for _, c in test])
    group = near_duplicates(x)
    cand = np.flatnonzero(group == np.arange(len(group)))
    texts = [test[i][0] for i in cand]
    labels, names = clusters(x[cand], texts)
    right = (pred[cand] == gold[cand]).astype(float)
    outlier = labels == -1
    weights = signal(conf[cand], outlier)
    truth = right.mean()
    rng = np.random.default_rng(seed)
    out = {"logs": len(test), "near_duplicates_dropped": int(len(test) - len(cand)), "candidates": int(len(cand)),
           "clusters": len(names), "outliers": int(outlier.sum()), "model_accuracy": float(truth),
           "failure_rate_outliers": float(1 - right[outlier].mean()), "failure_rate_clustered": float(1 - right[~outlier].mean()),
           "intents": int(len(set(gold))), "budget": budget, "reps": reps, "strategies": {}}
    for name in ("uniform", "stratified", "boosted"):
        rows = []
        for _ in range(reps):
            idx, pi = sample(name, budget, rng, strata=labels, weights=weights)
            rows.append({"n": len(idx), "failures": int((1 - right[idx]).sum()), "intents": len(set(gold[cand][idx])),
                         "clusters": len(set(labels[idx]) - {-1}), "error": estimate(right[idx], pi) - truth,
                         "naive_error": float(right[idx].mean() - truth)})
        agg = lambda k: float(np.mean([r[k] for r in rows]))
        out["strategies"][name] = {"cases": agg("n"), "failures": agg("failures"), "intents": agg("intents"),
                                   "clusters": agg("clusters"),
                                   "rmse": float(np.sqrt(np.mean([r["error"] ** 2 for r in rows]))),
                                   "bias": agg("error"), "naive_bias": agg("naive_error")}
    idx, pi = sample("boosted", budget, np.random.default_rng(seed), strata=labels, weights=weights)
    cases = [{"prompt": texts[i], "production_answer": str(pred[cand][i]), "expected": None,        # to be labelled
              "cluster": names.get(int(labels[i]), "outlier"), "confidence": round(float(conf[cand][i]), 3),
              "weight": round(float(1 / p), 2)} for i, p in zip(idx, pi)]
    out["version"] = write(cases, data.cache_dir() / "eval_set.jsonl", {"strategy": "boosted", "budget": budget, "seed": seed})
    out["examples"] = [dict(c, gold=str(gold[cand][i])) for c, i in zip(cases[:6], idx[:6])]
    out["cluster_names"] = [names[c] for c in sorted(names, key=lambda c: -(labels == c).sum())[:8]]
    return out


def bench():
    r = run()
    s = r["strategies"]
    lines = [f"{r['logs']:,} logged messages; {r['near_duplicates_dropped']} near-duplicates dropped; {r['clusters']} clusters found "
             f"(largest: {'; '.join(r['cluster_names'][:5])}), {r['outliers']} outliers. The production model is right on "
             f"{100 * r['model_accuracy']:.1f}%. Failure rate among outliers {100 * r['failure_rate_outliers']:.1f}%, "
             f"elsewhere {100 * r['failure_rate_clustered']:.1f}%.", "",
             f"Labelling budget {r['budget']} cases, {r['reps']} draws per strategy (means):", "",
             "| Strategy | Cases | Model failures captured | Intents covered (of 77) | Clusters covered | "
             "Accuracy estimate RMSE | Bias, weighted | Bias, unweighted mean |", "|---|---|---|---|---|---|---|---|"]
    for name, v in s.items():
        lines.append(f"| {name} | {v['cases']:.0f} | {v['failures']:.1f} | {v['intents']:.1f} | {v['clusters']:.1f} | "
                     f"{100 * v['rmse']:.2f} pts | {100 * v['bias']:+.2f} pts | {100 * v['naive_bias']:+.2f} pts |")
    lines += ["", f"A built set (boosted, version {r['version']}), first cases, with the gold label the labeller would add "
              "(Banking77, CC BY 4.0):", "", "| Message | Production answer | Gold | Cluster | Confidence | Weight |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {e['prompt']} | {e['production_answer']} | {e['gold']} | {e['cluster']} | {e['confidence']} | {e['weight']} |"
              for e in r["examples"]]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps({k: v for k, v in r.items() if k != "examples"}, indent=1))
    return "\n".join(lines)
