"""Banking77 (PolyAI, CC BY 4.0): 13,083 real customer-support messages to a bank, each with one
of 77 intents. Its test split stands in for a day of production logs; the train split trains
the production model whose answers the logs record. Downloaded on first use, with the
sentence embedder, into EVALGEN_DATA (default ~/.cache/evalgen), each pinned by SHA-256."""
import csv
import hashlib
import os
import tarfile
import time
import urllib.request
from pathlib import Path

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
FILES = {"banking77-train.csv": (f"{BANKING77}/train.csv", "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
         "banking77-test.csv": (f"{BANKING77}/test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
         "fast-bge-small-en-v1.5.tar.gz": ("https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
                                           "3858004b3822f64f940280874b8f2d2dc25b34a4f3eb3cdf617bdceeb21ed9ed")}


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
    target = cache_dir() / "fast-bge-small-en-v1.5"
    if not (target / "tokenizer.json").exists():
        with tarfile.open(fetch("fast-bge-small-en-v1.5.tar.gz")) as tar:
            tar.extractall(cache_dir(), members=[m for m in tar.getmembers() if not Path(m.name).name.startswith("._")],
                           filter="data")
    return target
