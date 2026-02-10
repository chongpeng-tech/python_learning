#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from threading import Lock
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖 fastapi，请先执行: pip install fastapi uvicorn"
    ) from exc

try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖 pydantic，请先执行: pip install pydantic"
    ) from exc

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖 sentence-transformers，请先执行: pip install sentence-transformers"
    ) from exc

from qa_sbert import (
    DEFAULT_RELATIONS,
    build_cypher,
    extract_answers_from_rows,
    extract_persons_by_alias,
    normalize_person_name,
)
from train_bert_bilstm_crf import (
    DEVICE_CHOICES,
    is_mps_available,
    load_model_for_inference,
    predict_sentence,
    resolve_device,
    resolve_input_path,
)

try:
    from py2neo import Graph
except Exception:  # pragma: no cover
    Graph = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"

app = FastAPI(title="红楼梦 NER + QA 控制台", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="static")

_cache_lock = Lock()
_ner_cache: dict[tuple[str, str], dict[str, Any]] = {}
_sbert_cache: dict[tuple[str, str], SentenceTransformer] = {}


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)
    ner_model_dir: str = "checkpoints/bert_bilstm_crf"
    device: str = "auto"


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    relations: list[str] = Field(default_factory=lambda: list(DEFAULT_RELATIONS))
    sbert_model: str = "BAAI/bge-small-zh-v1.5"
    ner_model_dir: str = "checkpoints/bert_bilstm_crf"
    device: str = "auto"
    run_neo4j: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"


def _resolve_torch_device(device_name: str):
    if device_name not in DEVICE_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 device: {device_name}，可选: {', '.join(DEVICE_CHOICES)}",
        )
    try:
        return resolve_device(device_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_ner_bundle(model_dir: str, device_name: str) -> dict[str, Any]:
    device = _resolve_torch_device(device_name)
    model_path = str(resolve_input_path(model_dir))
    key = (model_path, device.type)

    with _cache_lock:
        cached = _ner_cache.get(key)
        if cached is not None:
            return cached

    model, tokenizer, label2id, id2label, max_length = load_model_for_inference(
        model_dir=model_path,
        device=device,
    )
    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "max_length": max_length,
        "device": device,
        "model_dir": model_path,
    }

    with _cache_lock:
        _ner_cache[key] = bundle
    return bundle


def _extract_person_names(question: str, entities: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen = set()
    for entity in entities:
        if entity.get("type") != "PER":
            continue
        raw_name = entity.get("text", "").strip()
        if not raw_name:
            continue
        normalized = normalize_person_name(raw_name)
        if normalized not in seen:
            names.append(normalized)
            seen.add(normalized)

    if names:
        return names

    for alias_name in extract_persons_by_alias(question):
        if alias_name not in seen:
            names.append(alias_name)
            seen.add(alias_name)
    return names


def _get_sbert_model(model_name: str, device_name: str) -> SentenceTransformer:
    key = (model_name, device_name)
    with _cache_lock:
        model = _sbert_cache.get(key)
        if model is not None:
            return model

    try:
        model = SentenceTransformer(model_name, device=device_name)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="SBERT 模型加载失败，请检查网络连接或改用本地模型目录。",
        ) from exc

    with _cache_lock:
        _sbert_cache[key] = model
    return model


def _rank_relations(
    question: str,
    relations: list[str],
    sbert_model_name: str,
    device_name: str,
) -> tuple[str, list[dict[str, float | str]]]:
    sbert_model = _get_sbert_model(sbert_model_name, device_name)
    query_emb = sbert_model.encode(
        [question], convert_to_tensor=True, normalize_embeddings=True
    )
    rel_emb = sbert_model.encode(
        relations, convert_to_tensor=True, normalize_embeddings=True
    )
    scores_tensor = util.cos_sim(query_emb, rel_emb)[0]
    scores = [float(v) for v in scores_tensor.tolist()]

    ranking = [
        {"relation": rel, "score": score} for rel, score in zip(relations, scores)
    ]
    ranking.sort(key=lambda item: float(item["score"]), reverse=True)
    best_relation = str(ranking[0]["relation"])
    return best_relation, ranking


def _run_neo4j_query(
    uri: str,
    user: str,
    password: str,
    database: str,
    cypher: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    if Graph is None:
        raise RuntimeError("当前环境未安装 py2neo，请先 pip install py2neo")
    graph = Graph(uri, auth=(user, password), name=database)
    return graph.run(cypher, **params).data()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_ROOT / "index.html"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mps_available": is_mps_available(),
        "ner_cache_count": len(_ner_cache),
        "sbert_cache_count": len(_sbert_cache),
    }


@app.post("/api/ner")
def ner_predict(req: NERRequest) -> dict[str, Any]:
    bundle = _get_ner_bundle(req.ner_model_dir, req.device)

    token_label_pairs, entities = predict_sentence(
        sentence=req.text,
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        id2label=bundle["id2label"],
        max_length=bundle["max_length"],
        device=bundle["device"],
    )

    return {
        "text": req.text,
        "model_dir": bundle["model_dir"],
        "device": bundle["device"].type,
        "token_labels": [
            {"token": token, "label": label} for token, label in token_label_pairs
        ],
        "entities": entities,
    }


@app.post("/api/qa")
def qa_query(req: QARequest) -> dict[str, Any]:
    relations = [item.strip() for item in req.relations if item.strip()]
    if not relations:
        raise HTTPException(status_code=400, detail="relations 不能为空")

    bundle = _get_ner_bundle(req.ner_model_dir, req.device)
    token_label_pairs, entities = predict_sentence(
        sentence=req.question,
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        id2label=bundle["id2label"],
        max_length=bundle["max_length"],
        device=bundle["device"],
    )
    person_names = _extract_person_names(req.question, entities)

    best_relation, ranking = _rank_relations(
        question=req.question,
        relations=relations,
        sbert_model_name=req.sbert_model,
        device_name=bundle["device"].type,
    )
    cypher, params = build_cypher(person_names=person_names, relation=best_relation)

    neo4j_rows: list[dict[str, Any]] | None = None
    neo4j_error: str | None = None
    if req.run_neo4j:
        if not req.neo4j_password:
            neo4j_error = "未提供 Neo4j 密码，请填写 neo4j_password。"
        else:
            try:
                neo4j_rows = _run_neo4j_query(
                    uri=req.neo4j_uri,
                    user=req.neo4j_user,
                    password=req.neo4j_password,
                    database=req.neo4j_database,
                    cypher=cypher,
                    params=params,
                )
            except Exception as exc:
                neo4j_error = str(exc)
    else:
        neo4j_error = "未执行图谱查询，请勾选“直接查询 Neo4j”。"

    answers: list[str] = []
    answer_source = ""
    if neo4j_rows:
        answers = extract_answers_from_rows(neo4j_rows, person_names)
        if answers:
            answer_source = "neo4j"

    return {
        "question": req.question,
        "device": bundle["device"].type,
        "token_labels": [
            {"token": token, "label": label} for token, label in token_label_pairs
        ],
        "entities": entities,
        "person_names": person_names,
        "relation_ranking": ranking,
        "best_relation": best_relation,
        "cypher": cypher,
        "params": params,
        "answers": answers,
        "answer_source": answer_source,
        "neo4j_rows": neo4j_rows,
        "neo4j_error": neo4j_error,
    }


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动红楼梦 NER+QA 前端服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
