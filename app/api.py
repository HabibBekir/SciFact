"""API SciCheck : search + streaming generation, feedback, metrics."""

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.config import settings
from app.generation import generate_stream
from app.models import dense_model, qdrant, reranker, sparse_model
from app.monitoring import FEEDBACK, LATENCY, REQUESTS, TOP_SCORE, VERDICTS, parse_verdict, write_event
from app.search import retrieve

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # charge the models in the background, so that the first request is faster
    dense_model(), sparse_model()
    if settings.use_rerank:
        reranker()
    yield


app = FastAPI(title="SciCheck", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Feedback(BaseModel):
    request_id: str
    value: int = Field(ge=-1, le=1)  # 1 = useful, -1 = not useful
    comment: str = Field(default="", max_length=2000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    try:
        n = qdrant().count(settings.collection).count
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Qdrant indisponible : {e}")
    return {"status": "ok", "documents": n, "model": settings.llm_model}


@app.get("/api/search")
def api_search(q: str, k: int = 10):
    t0 = time.perf_counter()
    hits = retrieve(q)[:k]
    return {"latency_ms": round(1000 * (time.perf_counter() - t0)), "hits": hits}


@app.post("/api/ask")
def api_ask(req: AskRequest):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    hits = retrieve(req.question)
    t_retrieval = time.perf_counter() - t0
    LATENCY.labels("retrieval").observe(t_retrieval)
    if hits and "rerank_score" in hits[0]:
        TOP_SCORE.observe(hits[0]["rerank_score"])

    def stream():
        yield (
            json.dumps(
                {
                    "type": "sources",
                    "request_id": rid,
                    "sources": [
                        {
                            "doc_id": h["doc_id"],
                            "title": h["title"],
                            "text": h["text"][:400],
                            "score": round(h.get("rerank_score", h["score"]), 3),
                        }
                        for h in hits
                    ],
                }
            )
            + "\n"
        )
        answer, t_first, status = [], None, "ok"
        try:
            for token in generate_stream(req.question, hits):
                if t_first is None:
                    t_first = time.perf_counter() - t0
                    LATENCY.labels("ttft").observe(t_first)
                answer.append(token)
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except Exception as e:  # noqa: BLE001
            status = "llm_error"
            yield json.dumps({"type": "error", "message": f"Erreur LLM : {e}"}) + "\n"
        total = time.perf_counter() - t0
        LATENCY.labels("total").observe(total)
        REQUESTS.labels(status).inc()
        text = "".join(answer)
        verdict = parse_verdict(text)
        VERDICTS.labels(verdict).inc()
        yield json.dumps({"type": "done", "latency_s": round(total, 2)}) + "\n"
        write_event(
            {
                "event": "ask",
                "request_id": rid,
                "status": status,
                "question": req.question,
                "doc_ids": [h["doc_id"] for h in hits],
                "top_score": hits[0].get("rerank_score") if hits else None,
                "verdict": verdict,
                "answer": text,
                "model": settings.llm_model,
                "retrieval_s": round(t_retrieval, 3),
                "ttft_s": round(t_first, 3) if t_first else None,
                "total_s": round(total, 3),
            }
        )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/feedback")
def api_feedback(fb: Feedback):
    FEEDBACK.labels("up" if fb.value > 0 else "down").inc()
    write_event({"event": "feedback", **fb.model_dump()})
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
