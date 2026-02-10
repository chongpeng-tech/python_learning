#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import Any

from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from train_bert_bilstm_crf import (
    DEVICE_CHOICES,
    load_model_for_inference,
    predict_sentence,
    resolve_device,
    resolve_input_path,
)


DEFAULT_RELATIONS = ["父亲", "母亲", "住处", "丫鬟"]
ALIAS_MAP = {
    "宝玉": "贾宝玉",
    "黛玉": "林黛玉",
    "宝钗": "薛宝钗",
    "凤姐": "王熙凤",
}

SURNAME_HINTS = {"贾", "王", "薛", "林", "史", "秦", "尤", "李", "甄"}
NAME_BAD_PREFIXES = {
    "又",
    "便",
    "就",
    "将",
    "把",
    "并",
    "且",
    "而",
    "因",
    "向",
    "从",
    "于",
    "与",
    "这",
    "那",
    "其",
}
NAME_BAD_SUFFIX_CHARS = set("的了着呢吧呀啊嘛么者等来去上下里外中")
NAME_BAD_SUBSTRINGS = {
    "指着",
    "说道",
    "笑道",
    "问道",
    "听见",
    "看见",
    "告诉",
    "一个",
    "两个",
    "三个",
    "四个",
    "你们",
    "我们",
    "他们",
    "这里",
    "那里",
}


def normalize_person_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[的是在了呢吧呀啊嘛]+$", "", name)
    name = re.sub(r"^[的是在与和及]", "", name)
    return ALIAS_MAP.get(name, name)


def extract_persons_by_alias(question: str) -> list[str]:
    names: list[str] = []
    seen = set()

    # 优先匹配更长别名，减少短词误命中
    alias_items = sorted(ALIAS_MAP.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, canonical in alias_items:
        if alias in question or canonical in question:
            if canonical not in seen:
                names.append(canonical)
                seen.add(canonical)

    # 简单模式：例如“贾宝玉的父亲是谁”
    for match in re.finditer(r"[贾王薛林史秦尤][\u4e00-\u9fa5]{1,2}", question):
        candidate = match.group(0).strip()
        candidate = normalize_person_name(candidate)
        if candidate and candidate not in seen:
            names.append(candidate)
            seen.add(candidate)

    return names


def is_valid_answer_name(name: str) -> bool:
    if not name:
        return False
    if not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", name):
        return False
    if name[0] in NAME_BAD_PREFIXES:
        return False
    if name[-1] in NAME_BAD_SUFFIX_CHARS:
        return False
    if any(term in name for term in NAME_BAD_SUBSTRINGS):
        return False
    return True


def answer_name_prior(name: str) -> float:
    score = 0.0
    if name and name[0] in SURNAME_HINTS:
        score += 0.5
    if name.endswith(("夫人", "太君")):
        score += 0.2
    return score


def extract_answers_from_rows(
    rows: list[dict[str, Any]],
    person_names: list[str],
) -> list[str]:
    target_names = set(person_names)
    score_map: dict[str, float] = {}

    for row in rows:
        subject = str(row.get("subject", "")).strip()
        obj = str(row.get("object", "")).strip()
        try:
            edge_count = float(row.get("count", 1) or 1)
        except (TypeError, ValueError):
            edge_count = 1.0

        if target_names:
            if subject in target_names and obj:
                candidate = obj
            elif obj in target_names and subject:
                candidate = subject
            else:
                continue
        else:
            if obj:
                candidate = obj
            else:
                continue

        if candidate in target_names:
            continue
        if not is_valid_answer_name(candidate):
            continue

        score_map[candidate] = score_map.get(candidate, 0.0) + edge_count + answer_name_prior(
            candidate
        )

    ranked = sorted(
        score_map.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    return [name for name, _ in ranked]


def extract_persons_from_question(
    question: str,
    ner_model_dir: str,
    device: str,
) -> list[str]:
    torch_device = resolve_device(device)
    model, tokenizer, _label2id, id2label, max_length = load_model_for_inference(
        model_dir=ner_model_dir,
        device=torch_device,
    )
    _token_labels, entities = predict_sentence(
        sentence=question,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        max_length=max_length,
        device=torch_device,
    )
    names: list[str] = []
    seen = set()
    for entity in entities:
        if entity.get("type") != "PER":
            continue
        raw_name = entity.get("text", "").strip()
        if not raw_name:
            continue
        name = normalize_person_name(raw_name)
        if name not in seen:
            names.append(name)
            seen.add(name)

    if names:
        return names

    # NER 未命中时，回退到别名规则，保证“宝玉/黛玉”等可用
    return extract_persons_by_alias(question)


def relation_similarity(
    question: str,
    relations: list[str],
    sbert_model_name: str,
    device: str,
) -> tuple[str, list[tuple[str, float]]]:
    try:
        model = SentenceTransformer(sbert_model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            "SBERT 模型加载失败。请检查网络连接，或将 --sbert-model 指向本地已下载模型目录。"
        ) from exc
    query_emb = model.encode(
        [question], convert_to_tensor=True, normalize_embeddings=True
    )
    rel_emb = model.encode(
        relations, convert_to_tensor=True, normalize_embeddings=True
    )
    scores_tensor = util.cos_sim(query_emb, rel_emb)[0]
    scores = [float(v) for v in scores_tensor.tolist()]
    ranking = sorted(
        list(zip(relations, scores)),
        key=lambda item: item[1],
        reverse=True,
    )
    best_relation = ranking[0][0]
    return best_relation, ranking


def build_cypher(person_names: list[str], relation: str) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {"relation": relation}
    rel_filter = "(type(r) = $relation OR r.relation = $relation)"
    if not person_names:
        cypher = (
            "MATCH (p:Person)-[r]-(o) "
            f"WHERE {rel_filter} "
            "RETURN p.name AS subject, type(r) AS relation, o.name AS object, coalesce(r.count, 1) AS count LIMIT 20"
        )
        return cypher, params

    if len(person_names) == 1:
        cypher = (
            "MATCH (p:Person)-[r]-(o) "
            f"WHERE (p.name = $name OR o.name = $name) AND {rel_filter} "
            "RETURN p.name AS subject, type(r) AS relation, o.name AS object, coalesce(r.count, 1) AS count LIMIT 20"
        )
        params["name"] = person_names[0]
        return cypher, params

    cypher = (
        "MATCH (p:Person)-[r]-(o) "
        f"WHERE (p.name IN $names OR o.name IN $names) AND {rel_filter} "
        "RETURN p.name AS subject, type(r) AS relation, o.name AS object, coalesce(r.count, 1) AS count LIMIT 50"
    )
    params["names"] = person_names
    return cypher, params


def maybe_run_neo4j(
    run_neo4j: bool,
    uri: str,
    user: str,
    password: str | None,
    database: str,
    cypher: str,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not run_neo4j:
        return None, None

    if not password:
        raise ValueError(
            "未提供 Neo4j 密码，请使用 --neo4j-password 或设置 NEO4J_PASSWORD"
        )

    from py2neo import Graph

    try:
        graph = Graph(uri, auth=(user, password), name=database)
        cursor = graph.run(cypher, **params)
        rows = cursor.data()
        print("Neo4j 查询结果:")
        print(json.dumps(rows[:20], ensure_ascii=False, indent=2))
        if not rows:
            print("提示: 查询结果为空。若图谱当前只有 CO_OCCUR(共现)关系，父亲/母亲类查询会为空。")
        return rows, None
    except Exception as exc:
        print(f"Neo4j 查询失败: {exc}")
        print("请检查 Neo4j 服务是否启动、端口是否正确（默认 bolt://localhost:7687）。")
        return None, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SBERT 语义匹配关系 + NER 人名识别，生成 Cypher 查询语句"
    )
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_RELATIONS),
        help="标准关系列表，逗号分隔（默认: 父亲,母亲,住处,丫鬟）",
    )
    parser.add_argument(
        "--sbert-model",
        default="BAAI/bge-small-zh-v1.5",
        help="sentence-transformers 中文模型名称",
    )
    parser.add_argument(
        "--ner-model-dir",
        default="checkpoints/bert_bilstm_crf",
        help="NER 模型目录（相对路径按项目根目录解析）",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="推理设备，auto 优先 MPS",
    )

    parser.add_argument("--run-neo4j", action="store_true", help="是否直接执行 Cypher")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relations = [item.strip() for item in args.relations.split(",") if item.strip()]
    if not relations:
        raise ValueError("relations 不能为空")

    model_dir = str(resolve_input_path(args.ner_model_dir))
    device = resolve_device(args.device).type

    person_names = extract_persons_from_question(
        question=args.question,
        ner_model_dir=model_dir,
        device=device,
    )
    best_relation, ranking = relation_similarity(
        question=args.question,
        relations=relations,
        sbert_model_name=args.sbert_model,
        device=device,
    )

    cypher, params = build_cypher(person_names=person_names, relation=best_relation)

    print(f"问题: {args.question}")
    print(f"识别人名: {person_names if person_names else '未识别到'}")
    print("关系相似度:")
    for rel, score in ranking:
        print(f"- {rel}: {score:.4f}")
    print(f"最终关系: {best_relation}")
    print("Cypher:")
    print(cypher)
    print(f"参数: {json.dumps(params, ensure_ascii=False)}")

    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    neo4j_rows, neo4j_error = maybe_run_neo4j(
        run_neo4j=args.run_neo4j,
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=password,
        database=args.neo4j_database,
        cypher=cypher,
        params=params,
    )

    if not args.run_neo4j:
        print("最终答案: 未执行图谱查询（请添加 --run-neo4j）")
        return

    answers: list[str] = []
    if neo4j_rows:
        answers = extract_answers_from_rows(neo4j_rows, person_names)

    if answers:
        print(f"最终答案: {', '.join(answers)}")
        print("答案来源: neo4j")
    else:
        print("最终答案: 未找到明确实体答案")
        if neo4j_error:
            print(f"未命中原因: Neo4j 查询失败 ({neo4j_error})")
        else:
            print("未命中原因: 图谱中尚无该关系数据（请先运行自动关系抽取入图）")


if __name__ == "__main__":
    main()
