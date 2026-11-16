"""Banking77 (PolyAI, CC BY 4.0): 13,083 real customer-support messages to a bank, each with one
of 77 intents. Its test split stands in for a day of production logs; the train split trains
the production model whose answers the logs record. Downloaded on first use, with the
sentence embedder, into EVALGEN_DATA (default ~/.cache/evalgen), each pinned by SHA-256."""
import csv
import hashlib
import os
import time
import urllib.request
from pathlib import Path

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
FILES = {"banking77-train.csv": (f"{BANKING77}/train.csv", "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
         "banking77-test.csv": (f"{BANKING77}/test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d")}
# The sentence embedder: bge-small-en-v1.5 (BAAI, MIT) as the ONNX export fastembed uses, from
# Qdrant's Hugging Face repository at a pinned commit. Each file is checked against its SHA-256.
# (It used to come as one tarball from Qdrant's storage bucket, which no longer allows downloads.)
HF_FILE = "https://huggingface.co/{repo}/resolve/{rev}/{name}"
EMBEDDER = {"repo": "Qdrant/bge-small-en-v1.5-onnx-Q", "rev": "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
            "folder": "fast-bge-small-en-v1.5",
            "files": {"tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                      "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"}}


def cache_dir():
    path = Path(os.environ.get("EVALGEN_DATA", Path.home() / ".cache" / "evalgen"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(name, tries=4):
    url, sha = FILES[name]
    target = cache_dir() / name
    if not target.exists():
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    body = r.read()
                break
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        if hashlib.sha256(body).hexdigest() != sha:
            raise RuntimeError(f"{name}: the download does not match its pinned SHA-256")
        target.write_bytes(body)
    return target


def messages(split):
    with open(fetch(f"banking77-{split}.csv"), encoding="utf-8") as f:
        return [(row["text"].strip(), row["category"]) for row in csv.DictReader(f)]


def embedder_dir():
    return model_files(EMBEDDER)

def model_files(spec, tries=4):
    """The model's files in the cache, downloading any that are missing or do not match their SHA-256."""
    target = cache_dir() / spec["folder"]
    target.mkdir(parents=True, exist_ok=True)
    for name, sha in spec["files"].items():
        path = target / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            continue
        url = HF_FILE.format(repo=spec["repo"], rev=spec["rev"], name=name)
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    body = r.read()
                break
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        if hashlib.sha256(body).hexdigest() != sha:
            raise RuntimeError(f"{spec['repo']}/{name}: the download does not match its pinned SHA-256")
        path.write_bytes(body)
    return target
