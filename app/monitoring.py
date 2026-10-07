"""Structured logs (JSON Lines) + Prometheus metrics."""

import json
import logging
import time
from pathlib import Path

from prometheus_client import Counter, Histogram

from app.config import settings

REQUESTS = Counter("scicheck_requests_total", "Questions received", ["status"])
VERDICTS = Counter("scicheck_verdicts_total", "Verdicts produced", ["verdict"])
FEEDBACK = Counter("scicheck_feedback_total", "User feedback", ["value"])
LATENCY = Histogram(
    "scicheck_latency_seconds",
    "Latency by stage",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
TOP_SCORE = Histogram(
    "scicheck_top_rerank_score", "Score of the top reranked result", buckets=(-10, -5, -2, 0, 2, 4, 6, 8, 10)
)

log = logging.getLogger("scicheck")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def write_event(event: dict) -> None:
    """Adds an event to the JSONL file and to standard output."""
    event_dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    line = json.dumps(event_dict, ensure_ascii=False)
    log.info(line)
    path = Path(settings.log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_verdict(answer: str) -> str:
    a = answer.upper()
    for v in ("NOT ENOUGH INFO", "SUPPORTED", "REFUTED"):
        if v in a[:200]:
            return v
    return "OTHER"
