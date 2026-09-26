import json

import numpy as np

from evalgen.build import estimate, inclusion, near_duplicates, sample, signal, write
from evalgen.logs import ingest, redact


def test_redaction_and_luhn():
    text, counts = redact("Mail me at ann.lee@example.com, card 4111 1111 1111 1111, order 1234567890123, call +44 20 7946 0958")
    assert "[EMAIL]" in text and "[CARD]" in text and "[PHONE]" in text
    assert counts["card"] == 1 and "1234567890123" in text      # an order number: fails Luhn, has no phone separators
    assert redact("IBAN GB82 WEST 1234 5698 7654 32")[0] == "IBAN [IBAN]"


def test_ingest_validates_dedupes_and_redacts():
    lines = [json.dumps({"id": "1", "prompt": "hi, I'm bo@x.io", "response": "hello"}),
             json.dumps({"id": "2", "prompt": "hi, I'm bo@x.io", "response": "hello"}),
             json.dumps({"id": "3", "prompt": "", "response": "x"}), "not json"]
    entries, report = ingest(lines)
    assert [e.id for e in entries] == ["1"] and entries[0].prompt == "hi, I'm [EMAIL]"
    assert report == {"read": 4, "invalid": 2, "duplicates": 1, "redacted": {"email": 1}}


def test_near_duplicates_group_to_the_first():
    v = np.array([[1, 0], [0.999, 0.0447], [0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert list(near_duplicates(v)) == [0, 0, 2]


def test_boosted_sampling_is_unbiased_once_weighted():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.3, 1, 5000)
    right = (rng.random(5000) < confidence).astype(float)       # low confidence fails more
    w = signal(confidence, np.zeros(5000, bool))
    pi = inclusion(w, 200)
    assert abs(pi.sum() - 200) < 1e-6 and pi.max() <= 1
    est, naive, failures = [], [], []
    for _ in range(400):
        idx, p = sample("boosted", 200, rng, weights=w)
        est.append(estimate(right[idx], p))
        naive.append(right[idx].mean())
        failures.append((1 - right[idx]).sum())
    assert abs(np.mean(est) - right.mean()) < 0.005
    assert np.mean(naive) < right.mean() - 0.03
    uniform = [(1 - right[sample("uniform", 200, rng, strata=np.zeros(5000))[0]]).sum() for _ in range(100)]
    assert np.mean(failures) > 1.2 * np.mean(uniform)


def test_stratified_reaches_every_stratum_and_sets_are_versioned(tmp_path):
    strata = np.repeat([0, 1, 2], [990, 5, 5])
    hits = np.zeros(3)
    for seed in range(200):
        idx, _ = sample("stratified", 30, np.random.default_rng(seed), strata=strata)
        hits += [np.isin(s, strata[idx]) for s in range(3)]
    assert hits.min() > 100
    v1 = write([{"prompt": "a"}], tmp_path / "set.jsonl", {})
    assert v1 == write([{"prompt": "a"}], tmp_path / "set.jsonl", {}) != write([{"prompt": "b"}], tmp_path / "set.jsonl", {})
