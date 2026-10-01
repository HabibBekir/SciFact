"""Télécharge et décompresse BEIR SciFact dans data/scifact."""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEST = Path("data")

if (DEST / "scifact" / "corpus.jsonl").exists():
    print("SciFact déjà présent dans data/scifact")
else:
    print(f"Téléchargement de {URL} ...")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DEST)
    print("OK ->", sorted(p.name for p in (DEST / "scifact").iterdir()))
