#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import jieba
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖 jieba，请先执行: pip install jieba"
    ) from exc


DEFAULT_ENTITY_LEXICON = {
    "PER": {
        "贾宝玉",
        "林黛玉",
        "薛宝钗",
        "王熙凤",
        "贾母",
        "袭人",
        "晴雯",
        "史湘云",
        "贾政",
        "王夫人",
        "贾琏",
        "贾迎春",
        "贾探春",
        "贾惜春",
        "秦可卿",
        "妙玉",
        "香菱",
        "平儿",
        "鸳鸯",
        "李纨",
    },
    "LOC": {
        "大观园",
        "荣国府",
        "宁国府",
        "潇湘馆",
        "怡红院",
        "蘅芜苑",
        "稻香村",
        "栊翠庵",
        "金陵",
        "京师",
        "扬州",
        "姑苏",
        "长安",
    },
}
ENTITY_PRIORITY = ("PER", "LOC")
JIEBA_TAG_MAP = {"PER": "nr", "LOC": "ns"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\u3000", "").replace("\r", "").replace("\n", "")
    return [s.strip() for s in re.split(r"[。！!]+", normalized) if s.strip()]


def load_lexicon_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                names.add(item)
    return names


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_entity_lexicon(
    per_file: Path | None,
    loc_file: Path | None,
) -> dict[str, set[str]]:
    lexicon = {entity_type: set(values) for entity_type, values in DEFAULT_ENTITY_LEXICON.items()}
    lexicon["PER"].update(load_lexicon_file(per_file))
    lexicon["LOC"].update(load_lexicon_file(loc_file))
    return lexicon


def label_tokens(
    tokens: list[str],
    entity_lexicon: dict[str, set[str]],
    max_window: int,
) -> list[str]:
    if max_window <= 0:
        return ["O"] * len(tokens)

    labels = ["O"] * len(tokens)
    i = 0
    while i < len(tokens):
        matched_end = -1
        matched_type = None
        window_end = min(len(tokens), i + max_window)
        for j in range(window_end, i, -1):
            candidate = "".join(tokens[i:j])
            for entity_type in ENTITY_PRIORITY:
                if candidate in entity_lexicon.get(entity_type, set()):
                    matched_type = entity_type
                    matched_end = j
                    break
            if matched_end != -1:
                break

        if matched_end == -1:
            i += 1
            continue

        labels[i] = f"B-{matched_type}"
        for k in range(i + 1, matched_end):
            labels[k] = f"I-{matched_type}"
        i = matched_end
    return labels


def build_dataset(
    input_path: Path,
    output_path: Path,
    entity_lexicon: dict[str, set[str]],
) -> tuple[int, int]:
    for entity_type, names in entity_lexicon.items():
        jieba_tag = JIEBA_TAG_MAP.get(entity_type)
        for name in names:
            jieba.add_word(name, freq=1_000_000, tag=jieba_tag)

    text = input_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)

    max_window = max((len(name) for names in entity_lexicon.values() for name in names), default=0)
    token_count = 0
    sentence_count = 0

    with output_path.open("w", encoding="utf-8") as out:
        for sentence in sentences:
            tokens = [tok for tok in jieba.lcut(sentence) if tok.strip()]
            if not tokens:
                continue

            labels = label_tokens(tokens, entity_lexicon, max_window=max_window)
            for tok, label in zip(tokens, labels):
                out.write(f"{tok} {label}\n")
                token_count += 1
            out.write("\n")
            sentence_count += 1

    return sentence_count, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 jieba + 规则词表，生成 BERT-NER 伪标注数据。"
    )
    parser.add_argument(
        "--input",
        default="data/raw/honglou.txt",
        help="输入小说文本路径（默认相对项目根目录: data/raw/honglou.txt）",
    )
    parser.add_argument(
        "--output",
        default="data/processed/train.txt",
        help="输出 BERT-NER 训练数据路径（默认相对项目根目录: data/processed/train.txt）",
    )
    parser.add_argument(
        "--per-file",
        default=None,
        help="可选：额外人名词表，每行一个名字",
    )
    parser.add_argument(
        "--loc-file",
        default=None,
        help="可选：额外地名词表，每行一个名字",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(args.output)
    per_file = resolve_input_path(args.per_file) if args.per_file else None
    loc_file = resolve_input_path(args.loc_file) if args.loc_file else None

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    if per_file and not per_file.exists():
        raise FileNotFoundError(f"找不到人名文件: {per_file}")
    if loc_file and not loc_file.exists():
        raise FileNotFoundError(f"找不到地名文件: {loc_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entity_lexicon = build_entity_lexicon(per_file, loc_file)
    sentence_count, token_count = build_dataset(input_path, output_path, entity_lexicon)
    print(
        f"完成: 句子数={sentence_count}, token数={token_count}, 输出={output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
