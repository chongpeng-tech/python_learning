#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import torch

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    from py2neo import Graph
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少依赖 py2neo，请先执行: pip install py2neo") from exc

from train_bert_bilstm_crf import (
    DEVICE_CHOICES,
    load_model_for_inference,
    predict_sentence,
    resolve_device,
    resolve_input_path,
    resolve_output_path,
)

ALIAS_MAP = {
    "宝玉": "贾宝玉",
    "黛玉": "林黛玉",
    "宝钗": "薛宝钗",
    "凤姐": "王熙凤",
    "王氏": "王夫人",
    "政老爹": "贾政",
    "政老爷": "贾政",
    "政老": "贾政",
    "政公": "贾政",
    "赦老爹": "贾赦",
    "赦老": "贾赦",
    "珍爷": "贾珍",
    "敬老爹": "贾敬",
}

PERSON_STOPWORDS = {
    "老爷",
    "太太",
    "姑娘",
    "奶奶",
    "公子",
    "小姐",
    "媳妇",
    "婆子",
    "小厮",
    "奴才",
    "老太太",
}

FATHER_PATTERNS = [
    re.compile(r"(?P<child>[\u4e00-\u9fa5]{2,4})的父亲(?:是|为)?(?P<parent>[\u4e00-\u9fa5]{2,4})"),
    re.compile(r"(?P<parent>[\u4e00-\u9fa5]{2,4})是(?P<child>[\u4e00-\u9fa5]{2,4})的父亲"),
    re.compile(r"(?P<child>[\u4e00-\u9fa5]{2,4})之父(?P<parent>[\u4e00-\u9fa5]{2,4})"),
]

BIRTH_FATHER_PATTERNS = [
    re.compile(
        r"(?P<parent>[\u4e00-\u9fa5]{2,4})[^。．！？!?]{0,30}"
        r"(?:又生(?:了一位|了一个)?公子|生(?:了一位|了一个)?公子|生得一子|生了一子|生有一子)"
        r"[^。．！？!?]{0,30}(?:名唤|名叫|取名叫作|叫作)"
        r"(?P<child>[\u4e00-\u9fa5]{2,4})"
    ),
    re.compile(
        r"(?P<parent>[\u4e00-\u9fa5]{2,4})[^。．！？!?]{0,30}"
        r"(?:次子|二子|犬子|小犬)[^。．！？!?]{0,20}"
        r"(?:名唤|名叫|取名叫作|叫作)"
        r"(?P<child>[\u4e00-\u9fa5]{2,4})"
    ),
]

MOTHER_PATTERNS = [
    re.compile(r"(?P<child>[\u4e00-\u9fa5]{2,4})的母亲(?:是|为)?(?P<mother>[\u4e00-\u9fa5]{2,4})"),
    re.compile(r"(?P<mother>[\u4e00-\u9fa5]{2,4})是(?P<child>[\u4e00-\u9fa5]{2,4})的母亲"),
    re.compile(r"(?P<child>[\u4e00-\u9fa5]{2,4})之母(?P<mother>[\u4e00-\u9fa5]{2,4})"),
]

BIRTH_MOTHER_PATTERNS = [
    re.compile(
        r"(?P<mother>[\u4e00-\u9fa5]{1,3}(?:氏|夫人))[^。．！？!?]{0,30}"
        r"(?:头胎生(?:的)?公子|又生(?:了一位|了一个)?公子|生(?:了一位|了一个)?公子|生得一子)"
        r"[^。．！？!?]{0,30}(?:名唤|名叫|取名叫作|叫作)"
        r"(?P<child>[\u4e00-\u9fa5]{2,4})"
    ),
]

SPOUSE_CONTEXT_PATTERNS = [
    re.compile(r"(?P<father>[\u4e00-\u9fa5]{2,5})的夫人(?P<mother>[\u4e00-\u9fa5]{2,4})"),
    re.compile(r"(?P<mother>[\u4e00-\u9fa5]{2,4})是(?P<father>[\u4e00-\u9fa5]{2,5})的夫人"),
]

SPOUSE_BIRTH_PATTERNS = [
    re.compile(
        r"(?P<father>[\u4e00-\u9fa5]{2,5})的夫人(?P<mother>[\u4e00-\u9fa5]{2,4})"
        r"[^。．！？!?]{0,40}"
        r"(?:头胎生(?:的)?公子|又生(?:了一位|了一个)?公子|生(?:了一位|了一个)?公子|生得一子|生了一子|生有一子)"
        r"[^。．！？!?]{0,30}(?:名唤|名叫|取名叫作|叫作)"
        r"(?P<child>[\u4e00-\u9fa5]{2,4})"
    ),
]

CONTEXT_BIRTH_CHILD_PATTERNS = [
    re.compile(
        r"(?:头胎生(?:的)?公子|第二胎生(?:了一位|了一个)?(?:公子|小姐)|又生(?:了一位|了一个)?公子|生(?:了一位|了一个)?公子|生得一子|生了一子|生有一子)"
        r"[^。．！？!?]{0,80}(?:名唤|名叫|取名叫作|叫作)"
        r"(?P<child>[\u4e00-\u9fa5]{2,4})"
    ),
]

MAID_PATTERNS = [
    re.compile(r"(?P<owner>[\u4e00-\u9fa5]{2,4})的丫鬟(?:是|为)?(?P<maid>[\u4e00-\u9fa5]{2,4})"),
    re.compile(r"(?P<maid>[\u4e00-\u9fa5]{2,4})是(?P<owner>[\u4e00-\u9fa5]{2,4})的丫鬟"),
]

RESIDENCE_PATTERNS = [
    re.compile(r"(?P<person>[\u4e00-\u9fa5]{2,4})(?:住在|居住在|住于|居于)(?P<place>[\u4e00-\u9fa5]{2,10})"),
    re.compile(r"(?P<place>[\u4e00-\u9fa5]{2,10})是(?P<person>[\u4e00-\u9fa5]{2,4})的住处"),
]

RESIDENCE_HINTS = ("住在", "居住", "居于", "住于", "住处")


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\u3000", "").replace("\r", "")
    return [s.strip() for s in re.split(r"[。．！？!?；;\n]+", normalized) if s.strip()]


def read_sentences(input_path: Path) -> list[str]:
    text = input_path.read_text(encoding="utf-8")
    return split_sentences(text)


def normalize_person_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    suffix_chars = "的是在了呢吧呀啊嘛说道见听命因先后又便来去忙拿向与及者等自"
    prefix_chars = "的是在与和及这那其"
    while len(name) > 2 and name and name[-1] in suffix_chars:
        name = name[:-1]
    while len(name) > 2 and name and name[0] in prefix_chars:
        name = name[1:]
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    for ending in ("老爷", "老爹"):
        if len(name) > 2 and name.endswith(ending):
            name = name[: -len(ending)]
            break
    return ALIAS_MAP.get(name, name)


def unique_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in values:
        value = item.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def is_plausible_person_name(name: str) -> bool:
    if not name:
        return False
    if len(name) < 2 or len(name) > 4:
        return False
    if name in PERSON_STOPWORDS:
        return False
    if any(ch in "，。；：、！？“”\"'（）()《》【】[] " for ch in name):
        return False
    return True


def match_candidate(
    raw: str,
    candidates: list[str],
    normalize_fn,
    allow_fallback: bool = False,
    fallback_validator: Callable[[str], bool] | None = None,
) -> str | None:
    if not raw:
        return None
    raw_norm = normalize_fn(raw.strip())
    if raw_norm in candidates:
        return raw_norm
    for cand in candidates:
        if raw_norm in cand or cand in raw_norm:
            return cand
    if allow_fallback and raw_norm:
        if fallback_validator is None or fallback_validator(raw_norm):
            return raw_norm
    return None


def extract_entities_from_sentence(
    sentence: str,
    model,
    tokenizer,
    id2label: dict[int, str],
    max_length: int,
    device: torch.device,
    min_name_length: int,
) -> tuple[list[str], list[str]]:
    _token_labels, entities = predict_sentence(
        sentence=sentence,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        max_length=max_length,
        device=device,
    )

    persons: list[str] = []
    locations: list[str] = []
    for entity in entities:
        text = str(entity.get("text", "")).strip()
        if len(text) < min_name_length:
            continue
        entity_type = str(entity.get("type", "")).upper()
        if entity_type == "PER":
            if len(text) > 6:
                continue
            normalized = normalize_person_name(text)
            if not normalized or normalized in PERSON_STOPWORDS:
                continue
            persons.append(normalized)
        elif entity_type == "LOC":
            if len(text) > 12:
                continue
            locations.append(text)

    return unique_keep_order(persons), unique_keep_order(locations)


def add_relation(
    triples: set[tuple[str, str, str]],
    subject: str | None,
    relation: str,
    obj: str | None,
) -> None:
    if not subject or not obj:
        return
    subject = subject.strip()
    obj = obj.strip()
    if not subject or not obj or subject == obj:
        return
    triples.add((subject, relation, obj))


def extract_family_relations(
    sentence: str,
    persons: list[str],
    parent_context: tuple[str, str] | None = None,
) -> tuple[set[tuple[str, str, str]], tuple[str, str] | None]:
    triples: set[tuple[str, str, str]] = set()
    detected_parent_context: tuple[str, str] | None = None

    for pattern in SPOUSE_BIRTH_PATTERNS:
        for match in pattern.finditer(sentence):
            father = match_candidate(
                match.group("father"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            mother = match_candidate(
                match.group("mother"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            child = match_candidate(
                match.group("child"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            add_relation(triples, child, "父亲", father)
            add_relation(triples, child, "母亲", mother)
            if father and mother:
                detected_parent_context = (father, mother)

    for pattern in SPOUSE_CONTEXT_PATTERNS:
        for match in pattern.finditer(sentence):
            father = match_candidate(
                match.group("father"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            mother = match_candidate(
                match.group("mother"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            if father and mother:
                detected_parent_context = (father, mother)

    for pattern in FATHER_PATTERNS:
        for match in pattern.finditer(sentence):
            child = match_candidate(
                match.group("child"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            parent = match_candidate(
                match.group("parent"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            add_relation(triples, child, "父亲", parent)

    for pattern in MOTHER_PATTERNS:
        for match in pattern.finditer(sentence):
            child = match_candidate(
                match.group("child"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            mother = match_candidate(
                match.group("mother"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            add_relation(triples, child, "母亲", mother)

    for pattern in MAID_PATTERNS:
        for match in pattern.finditer(sentence):
            owner = match_candidate(match.group("owner"), persons, normalize_person_name)
            maid = match_candidate(match.group("maid"), persons, normalize_person_name)
            add_relation(triples, owner, "丫鬟", maid)

    for pattern in BIRTH_FATHER_PATTERNS:
        for match in pattern.finditer(sentence):
            parent = match_candidate(
                match.group("parent"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            child = match_candidate(
                match.group("child"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            add_relation(triples, child, "父亲", parent)

    for pattern in BIRTH_MOTHER_PATTERNS:
        for match in pattern.finditer(sentence):
            mother = match_candidate(
                match.group("mother"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            child = match_candidate(
                match.group("child"),
                persons,
                normalize_person_name,
                allow_fallback=True,
                fallback_validator=is_plausible_person_name,
            )
            add_relation(triples, child, "母亲", mother)

    active_context = detected_parent_context or parent_context
    if active_context:
        father, mother = active_context
        for pattern in CONTEXT_BIRTH_CHILD_PATTERNS:
            for match in pattern.finditer(sentence):
                child = match_candidate(
                    match.group("child"),
                    persons,
                    normalize_person_name,
                    allow_fallback=True,
                    fallback_validator=is_plausible_person_name,
                )
                add_relation(triples, child, "父亲", father)
                add_relation(triples, child, "母亲", mother)

    # 轻量兜底：句子中出现关键词且识别出 >=2 个人名，则按出现顺序取前两个
    if not triples and len(persons) >= 2:
        if "父亲" in sentence:
            add_relation(triples, persons[0], "父亲", persons[1])
        if "母亲" in sentence:
            add_relation(triples, persons[0], "母亲", persons[1])
        if "丫鬟" in sentence:
            add_relation(triples, persons[0], "丫鬟", persons[1])

    return triples, detected_parent_context


def extract_residence_relations(
    sentence: str,
    persons: list[str],
    locations: list[str],
) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()

    for pattern in RESIDENCE_PATTERNS:
        for match in pattern.finditer(sentence):
            person = match_candidate(match.group("person"), persons, normalize_person_name)
            place = match_candidate(match.group("place"), locations, lambda x: x)
            add_relation(triples, person, "住处", place)

    # 兜底：出现居住关键词且句内有人名+地名
    if not triples and any(hint in sentence for hint in RESIDENCE_HINTS):
        if persons and locations:
            add_relation(triples, persons[0], "住处", locations[0])

    return triples


def extract_relation_triples_from_sentence(
    sentence: str,
    persons: list[str],
    locations: list[str],
    parent_context: tuple[str, str] | None = None,
) -> tuple[set[tuple[str, str, str]], tuple[str, str] | None]:
    triples: set[tuple[str, str, str]] = set()
    family_triples, detected_parent_context = extract_family_relations(
        sentence,
        persons,
        parent_context=parent_context,
    )
    triples.update(family_triples)
    triples.update(extract_residence_relations(sentence, persons, locations))
    return triples, detected_parent_context


def analyze_sentences(
    sentences: list[str],
    model,
    tokenizer,
    id2label: dict[int, str],
    max_length: int,
    device: torch.device,
    min_name_length: int,
    progress_step: int,
    enable_cooccur: bool,
    enable_semantic_relations: bool,
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str, str]]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    relation_counts: Counter[tuple[str, str, str]] = Counter()
    parent_context: tuple[str, str] | None = None
    parent_context_ttl = 0

    for idx, sentence in enumerate(sentences, start=1):
        persons, locations = extract_entities_from_sentence(
            sentence=sentence,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            max_length=max_length,
            device=device,
            min_name_length=min_name_length,
        )

        if enable_cooccur and len(persons) >= 2:
            for left, right in combinations(sorted(set(persons)), 2):
                pair_counts[(left, right)] += 1

        if enable_semantic_relations:
            triples, detected_parent_context = extract_relation_triples_from_sentence(
                sentence,
                persons,
                locations,
                parent_context=parent_context if parent_context_ttl > 0 else None,
            )
            for triple in triples:
                relation_counts[triple] += 1
            if detected_parent_context:
                parent_context = detected_parent_context
                parent_context_ttl = 2
            elif parent_context_ttl > 0:
                parent_context_ttl -= 1
                if parent_context_ttl == 0:
                    parent_context = None

        if progress_step > 0 and idx % progress_step == 0:
            print(f"已处理句子 {idx}/{len(sentences)}")

    return pair_counts, relation_counts


def write_cooccur_csv(output_path: Path, pair_counts: Counter[tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person1", "relation", "person2", "count"])
        for (person1, person2), count in sorted(
            pair_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        ):
            writer.writerow([person1, "共现", person2, count])


def write_relation_csv(
    output_path: Path,
    relation_counts: Counter[tuple[str, str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "relation", "object", "count"])
        for (subject, relation, obj), count in sorted(
            relation_counts.items(),
            key=lambda item: (-item[1], item[0][1], item[0][0], item[0][2]),
        ):
            writer.writerow([subject, relation, obj, count])


def write_to_neo4j(
    pair_counts: Counter[tuple[str, str]],
    relation_counts: Counter[tuple[str, str, str]],
    uri: str,
    user: str,
    password: str,
    database: str,
    clear_existing: bool,
) -> None:
    graph = Graph(uri, auth=(user, password), name=database)

    if clear_existing:
        graph.run("MATCH ()-[r:CO_OCCUR|RELATION]->() DELETE r")

    cooccur_rows = [
        {"person1": person1, "person2": person2, "count": count}
        for (person1, person2), count in pair_counts.items()
    ]

    if cooccur_rows:
        graph.run(
            """
            UNWIND $rows AS row
            MERGE (p1:Person {name: row.person1})
            MERGE (p2:Person {name: row.person2})
            MERGE (p1)-[r:CO_OCCUR]->(p2)
            SET r.relation = '共现', r.count = row.count
            """,
            rows=cooccur_rows,
        )

    person_rows = [
        {"subject": s, "relation": r, "object": o, "count": c}
        for (s, r, o), c in relation_counts.items()
        if r != "住处"
    ]
    location_rows = [
        {"subject": s, "relation": r, "object": o, "count": c}
        for (s, r, o), c in relation_counts.items()
        if r == "住处"
    ]

    if person_rows:
        graph.run(
            """
            UNWIND $rows AS row
            MERGE (s:Person {name: row.subject})
            MERGE (o:Person {name: row.object})
            MERGE (s)-[rel:RELATION {relation: row.relation}]->(o)
            SET rel.count = row.count
            """,
            rows=person_rows,
        )

    if location_rows:
        graph.run(
            """
            UNWIND $rows AS row
            MERGE (s:Person {name: row.subject})
            MERGE (o:Location {name: row.object})
            MERGE (s)-[rel:RELATION {relation: row.relation}]->(o)
            SET rel.count = row.count
            """,
            rows=location_rows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练好的 BERT-BiLSTM-CRF 模型自动抽取关系并写入 Neo4j"
    )
    parser.add_argument(
        "--input",
        default="data/raw/honglou.txt",
        help="输入小说路径（相对路径按项目根目录解析）",
    )
    parser.add_argument(
        "--model-dir",
        default="checkpoints/bert_bilstm_crf",
        help="训练好的模型目录（相对路径按项目根目录解析）",
    )
    parser.add_argument(
        "--triples-path",
        default="triples.csv",
        help="输出共现 triples.csv 路径（相对路径按项目根目录解析）",
    )
    parser.add_argument(
        "--relation-triples-path",
        default="relation_triples.csv",
        help="输出语义关系 triples.csv 路径（相对路径按项目根目录解析）",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="推理设备，auto 优先 MPS",
    )
    parser.add_argument("--min-name-length", type=int, default=2, help="实体最短长度")
    parser.add_argument("--progress-step", type=int, default=500, help="进度打印步长")

    parser.add_argument("--skip-cooccur", action="store_true", help="不抽取共现关系")
    parser.add_argument(
        "--skip-semantic-relations",
        action="store_true",
        help="不抽取父亲/母亲/住处/丫鬟关系",
    )

    parser.add_argument("--skip-neo4j", action="store_true", help="仅生成 CSV，不写 Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="写入前清空已有 CO_OCCUR 与 RELATION 关系",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    model_dir = resolve_input_path(args.model_dir)
    triples_path = resolve_output_path(args.triples_path)
    relation_triples_path = resolve_output_path(args.relation_triples_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {model_dir}")

    enable_cooccur = not args.skip_cooccur
    enable_semantic_relations = not args.skip_semantic_relations
    if not enable_cooccur and not enable_semantic_relations:
        raise ValueError("至少要启用一种抽取：去掉 --skip-cooccur 或 --skip-semantic-relations")

    device = resolve_device(args.device)

    model, tokenizer, _label2id, id2label, max_length = load_model_for_inference(
        model_dir=str(model_dir),
        device=device,
    )
    sentences = read_sentences(input_path)
    print(f"句子总数: {len(sentences)} | 推理设备: {device.type}")

    pair_counts, relation_counts = analyze_sentences(
        sentences=sentences,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        max_length=max_length,
        device=device,
        min_name_length=args.min_name_length,
        progress_step=args.progress_step,
        enable_cooccur=enable_cooccur,
        enable_semantic_relations=enable_semantic_relations,
    )

    if enable_cooccur:
        write_cooccur_csv(triples_path, pair_counts)
        print(f"已写出共现 triples: {len(pair_counts)} 条 -> {triples_path}")

    if enable_semantic_relations:
        write_relation_csv(relation_triples_path, relation_counts)
        print(f"已写出语义 triples: {len(relation_counts)} 条 -> {relation_triples_path}")

    if args.skip_neo4j:
        print("已跳过 Neo4j 写入。")
        return

    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "未提供 Neo4j 密码，请使用 --neo4j-password 或设置环境变量 NEO4J_PASSWORD"
        )

    try:
        write_to_neo4j(
            pair_counts=pair_counts,
            relation_counts=relation_counts,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
            clear_existing=args.clear_existing,
        )
        print("已写入 Neo4j。")
    except Exception as exc:
        print(f"Neo4j 写入失败: {exc}")
        print("请检查 Neo4j 服务是否启动、端口是否正确（默认 bolt://localhost:7687）。")


if __name__ == "__main__":
    main()
