"""Chargement d'un jeu de données au format BEIR."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from app.config import settings


def load_corpus(data_dir: str = settings.data_dir) -> dict[str, dict]:
    corpus = {}
    with open(Path(data_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title", ""), "text": d["text"]}
    return corpus


def load_queries(data_dir: str = settings.data_dir) -> dict[str, str]:
    queries = {}
    with open(Path(data_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    return queries


def load_qrels(data_dir: str = settings.data_dir, split: str = "test") -> dict[str, dict[str, int]]:
    """qrels[query_id][doc_id] = score de pertinence (1 = pertinent)."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(Path(data_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # en-tête
        for qid, did, score in reader:
            qrels[str(qid)][str(did)] = int(score)
    return dict(qrels)


def doc_to_text(doc: dict) -> str:
    """Texte indexé : titre + résumé."""
    return f"{doc['title']}. {doc['text']}".strip()
