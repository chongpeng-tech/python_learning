#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

try:
    from torchcrf import CRF as _CRFImpl
    _CRF_BACKEND = "pytorch-crf"
except ImportError:
    try:
        from TorchCRF import CRF as _CRFImpl
        _CRF_BACKEND = "TorchCRF"
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "缺少 CRF 依赖，请安装任意一个: pip install pytorch-crf 或 pip install TorchCRF"
        ) from exc

try:
    import jieba
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖 jieba，请先执行: pip install jieba"
    ) from exc


EXPECTED_ENTITY_TYPES = ("PER", "LOC")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


class CRFWrapper(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        init_params = inspect.signature(_CRFImpl.__init__).parameters
        init_kwargs: dict[str, Any] = {}

        if "batch_first" in init_params:
            init_kwargs["batch_first"] = True
        if "use_gpu" in init_params:
            init_kwargs["use_gpu"] = torch.cuda.is_available()

        self.impl = _CRFImpl(num_labels, **init_kwargs)
        self._supports_reduction = "reduction" in inspect.signature(self.impl.forward).parameters
        self._decode_method = (
            "decode"
            if hasattr(self.impl, "decode")
            else "viterbi_decode"
            if hasattr(self.impl, "viterbi_decode")
            else ""
        )
        if not self._decode_method:
            raise RuntimeError("当前 CRF 实现不支持 decode/viterbi_decode")

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        if self._supports_reduction:
            log_likelihood = self.impl(emissions, labels, mask=mask, reduction="mean")
        else:
            log_likelihood = self.impl(emissions, labels, mask=mask)
        return -log_likelihood.mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        decode_fn = getattr(self.impl, self._decode_method)
        return decode_fn(emissions, mask=mask.bool())


@dataclass
class Feature:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    crf_mask: list[int]


class NERDataset(Dataset):
    def __init__(
        self,
        sentences: list[list[str]],
        tags: list[list[str]],
        tokenizer: BertTokenizerFast,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.features: list[Feature] = []
        o_id = label2id["O"]
        for tokens, labels in zip(sentences, tags):
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            word_ids = encoded.word_ids()
            if word_ids is None:
                continue

            label_ids: list[int] = []
            crf_mask: list[int] = []
            prev_word_idx: int | None = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(o_id)
                    crf_mask.append(0)
                    prev_word_idx = None
                    continue

                raw_label = labels[word_idx]
                aligned_label = (
                    raw_label if word_idx != prev_word_idx else to_inside_label(raw_label)
                )
                label_ids.append(label2id.get(aligned_label, o_id))
                crf_mask.append(1)
                prev_word_idx = word_idx

            self.features.append(
                Feature(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=label_ids,
                    crf_mask=crf_mask,
                )
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Feature:
        return self.features[idx]


class BertBiLSTMCRF(nn.Module):
    def __init__(
        self,
        bert_model_name_or_path: str,
        num_labels: int,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRFWrapper(num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crf_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decode: bool = False,
    ) -> tuple[torch.Tensor | None, list[list[int]] | None]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        lstm_output, _ = self.bilstm(sequence_output)
        emissions = self.classifier(self.dropout(lstm_output))

        loss = None
        if labels is not None:
            loss = self.crf.neg_log_likelihood(emissions, labels, crf_mask)

        predictions = self.crf.decode(emissions, crf_mask) if decode else None
        return loss, predictions


def to_inside_label(label: str) -> str:
    if label.startswith("B-"):
        return f"I-{label[2:]}"
    return label


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device_preference: str = "auto") -> torch.device:
    if device_preference == "auto":
        if is_mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_preference == "mps":
        if not is_mps_available():
            raise ValueError("当前环境不支持 MPS，请先检查 PyTorch 安装或改用 --device cpu")
        return torch.device("mps")

    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境不支持 CUDA，请改用 --device mps 或 --device cpu")
        return torch.device("cuda")

    if device_preference == "cpu":
        return torch.device("cpu")

    raise ValueError(f"不支持的设备类型: {device_preference}")


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


def read_bio_file(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    sentences: list[list[str]] = []
    tags: list[list[str]] = []
    current_tokens: list[str] = []
    current_tags: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
                continue

            parts = line.split()
            if len(parts) != 2:
                continue
            token, label = parts
            current_tokens.append(token)
            current_tags.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        tags.append(current_tags)

    return sentences, tags


def build_label_maps(
    tags: list[list[str]],
    expected_entity_types: tuple[str, ...] = EXPECTED_ENTITY_TYPES,
) -> tuple[dict[str, int], dict[int, str], list[str]]:
    observed_labels = {label for seq in tags for label in seq}
    observed_labels.add("O")

    ordered_labels = ["O"]
    for entity_type in expected_entity_types:
        ordered_labels.append(f"B-{entity_type}")
        ordered_labels.append(f"I-{entity_type}")

    for label in sorted(observed_labels):
        if label not in ordered_labels:
            ordered_labels.append(label)

    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    missing = [label for label in ordered_labels if label not in observed_labels and label != "O"]
    return label2id, id2label, missing


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[Feature]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item.input_ids, dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item.attention_mask, dtype=torch.long) for item in batch]
        labels = [torch.tensor(item.labels, dtype=torch.long) for item in batch]
        crf_mask = [torch.tensor(item.crf_mask, dtype=torch.bool) for item in batch]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=0),
            "crf_mask": pad_sequence(crf_mask, batch_first=True, padding_value=False),
        }

    return collate_fn


@torch.no_grad()
def evaluate(
    model: BertBiLSTMCRF,
    dataloader: DataLoader,
    o_id: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    tp = 0
    fp = 0
    fn = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        crf_mask = batch["crf_mask"].to(device)

        loss, predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crf_mask=crf_mask,
            labels=labels,
            decode=True,
        )
        if loss is not None:
            total_loss += loss.item()
            steps += 1

        assert predictions is not None
        for idx, pred_seq in enumerate(predictions):
            gold_seq = labels[idx][crf_mask[idx]].tolist()
            for pred_id, gold_id in zip(pred_seq, gold_seq):
                if pred_id == gold_id and gold_id != o_id:
                    tp += 1
                else:
                    if pred_id != o_id:
                        fp += 1
                    if gold_id != o_id:
                        fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_loss = total_loss / max(steps, 1)
    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_artifacts(
    output_dir: Path,
    model: BertBiLSTMCRF,
    tokenizer: BertTokenizerFast,
    label2id: dict[str, int],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bert_dir = output_dir / "bert"
    model.bert.save_pretrained(bert_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    with (output_dir / "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def train(
    train_file: str = "data/processed/train.txt",
    bert_model_name_or_path: str = "bert-base-chinese",
    output_dir: str = "checkpoints/bert_bilstm_crf",
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    head_learning_rate: float | None = None,
    weight_decay: float = 0.01,
    valid_ratio: float = 0.1,
    warmup_ratio: float = 0.1,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 1,
    dropout: float = 0.1,
    seed: int = 42,
    device_preference: str = "auto",
    log_steps: int = 20,
    gradient_accumulation_steps: int = 1,
    early_stop_patience: int = 0,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = False,
    mps_empty_cache_steps: int = 0,
) -> None:
    set_seed(seed)
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须 >= 1")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience 不能为负数")
    if mps_empty_cache_steps < 0:
        raise ValueError("mps_empty_cache_steps 不能为负数")

    train_path = resolve_input_path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练文件: {train_path}")

    sentences, tags = read_bio_file(train_path)
    if not sentences:
        raise ValueError(f"训练文件为空或格式不正确: {train_path}")

    label2id, id2label, missing_labels = build_label_maps(tags)
    if missing_labels:
        missing_begin_labels = [label for label in missing_labels if label.startswith("B-")]
        if missing_begin_labels:
            print(
                "警告: 训练集缺少以下标签样本，模型难以学习对应实体: "
                + ", ".join(missing_labels)
            )
        else:
            print(
                "提示: 当前训练集缺少 I-* 标签，常见于规则分词把实体切成单 token 的情况。"
            )

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_or_path)
    dataset = NERDataset(
        sentences=sentences,
        tags=tags,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    if len(dataset) < 2:
        raise ValueError("数据量过小，至少需要 2 句数据用于训练。")

    valid_size = max(1, int(len(dataset) * valid_ratio))
    if valid_size >= len(dataset):
        valid_size = len(dataset) - 1
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed),
    )

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = resolve_device(device_preference)
    print(f"CRF后端: {_CRF_BACKEND}")
    if device.type == "mps" and _CRF_BACKEND == "TorchCRF":
        print("提示: 当前使用 TorchCRF 后端，若 MPS 训练不稳定，建议安装 pytorch-crf。")

    model = BertBiLSTMCRF(
        bert_model_name_or_path=bert_model_name_or_path,
        num_labels=len(label2id),
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
    ).to(device)
    if gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()
        print("已启用 gradient checkpointing（更省显存，训练更慢）。")

    if head_learning_rate is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        no_decay_terms = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        bert_named = []
        head_named = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert."):
                bert_named.append((name, param))
            else:
                head_named.append((name, param))

        def build_param_groups(
            named_params: list[tuple[str, nn.Parameter]],
            lr: float,
            decay: float,
        ) -> list[dict[str, Any]]:
            decay_params = [
                p for n, p in named_params if not any(term in n for term in no_decay_terms)
            ]
            no_decay_params = [
                p for n, p in named_params if any(term in n for term in no_decay_terms)
            ]
            groups: list[dict[str, Any]] = []
            if decay_params:
                groups.append({"params": decay_params, "lr": lr, "weight_decay": decay})
            if no_decay_params:
                groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
            return groups

        optimizer_groups = []
        optimizer_groups.extend(
            build_param_groups(bert_named, lr=learning_rate, decay=weight_decay)
        )
        optimizer_groups.extend(
            build_param_groups(head_named, lr=head_learning_rate, decay=weight_decay)
        )
        optimizer = torch.optim.AdamW(optimizer_groups)

    update_steps_per_epoch = max(math.ceil(len(train_loader) / gradient_accumulation_steps), 1)
    total_steps = max(update_steps_per_epoch * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    o_id = label2id["O"]
    best_f1 = -1.0
    epochs_without_improve = 0
    save_dir = resolve_output_path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"开始训练: train={len(train_dataset)}, valid={len(valid_dataset)}, device={device.type}"
    )
    if device.type == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        print(
            "提示: 可设置 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 解除 MPS 显存上限（有系统不稳定风险）。"
        )
    print(
        f"训练配置: lr={learning_rate}, head_lr={head_learning_rate or learning_rate}, "
        f"weight_decay={weight_decay}, grad_accum={gradient_accumulation_steps}, "
        f"warmup_steps={warmup_steps}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        total_train_steps = len(train_loader)
        print(f"Epoch {epoch}/{epochs} 开始，batch 数={total_train_steps}")
        print("提示: 首个 step 在 MPS 上可能需要较长初始化时间。")
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            crf_mask = batch["crf_mask"].to(device)

            try:
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    crf_mask=crf_mask,
                    labels=labels,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "mps":
                    if hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                    raise RuntimeError(
                        "MPS 显存不足。建议减小 --batch-size/--max-length，"
                        "并增大 --gradient-accumulation-steps，或开启 --gradient-checkpointing。"
                    ) from exc
                raise
            if loss is None:
                continue
            raw_loss = loss.item()
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += raw_loss

            should_update = (
                step % gradient_accumulation_steps == 0 or step == total_train_steps
            )
            if should_update:
                if max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if (
                    device.type == "mps"
                    and mps_empty_cache_steps > 0
                    and step % mps_empty_cache_steps == 0
                    and hasattr(torch, "mps")
                ):
                    torch.mps.empty_cache()

            if step == 1 or step == total_train_steps or (log_steps > 0 and step % log_steps == 0):
                elapsed = time.perf_counter() - epoch_start
                speed = step / elapsed if elapsed > 0 else 0.0
                eta = (total_train_steps - step) / max(speed, 1e-8)
                print(
                    f"Epoch {epoch}/{epochs} step {step}/{total_train_steps} "
                    f"| loss={raw_loss:.4f} | {speed:.2f} step/s | eta={eta/60:.1f} min"
                )

        train_loss = epoch_loss / max(len(train_loader), 1)
        metrics = evaluate(model, valid_loader, o_id=o_id, device=device)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"| valid_loss={metrics['loss']:.4f} | P={metrics['precision']:.4f} "
            f"| R={metrics['recall']:.4f} | F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            config = {
                "bert_model_name_or_path": bert_model_name_or_path,
                "max_length": max_length,
                "lstm_hidden_size": lstm_hidden_size,
                "lstm_layers": lstm_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "head_learning_rate": head_learning_rate or learning_rate,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "gradient_checkpointing": gradient_checkpointing,
                "mps_empty_cache_steps": mps_empty_cache_steps,
                "warmup_ratio": warmup_ratio,
                "valid_ratio": valid_ratio,
                "expected_entity_types": list(EXPECTED_ENTITY_TYPES),
                "id2label": {str(k): v for k, v in id2label.items()},
            }
            save_training_artifacts(
                output_dir=save_dir,
                model=model,
                tokenizer=tokenizer,
                label2id=label2id,
                config=config,
            )
            print(f"已保存当前最优模型, F1={best_f1:.4f}, 路径={save_dir.resolve()}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(
                    f"触发早停: 连续 {epochs_without_improve} 个 epoch 未提升 F1，训练结束。"
                )
                break


def load_model_for_inference(
    model_dir: str,
    device: torch.device | None = None,
) -> tuple[BertBiLSTMCRF, BertTokenizerFast, dict[str, int], dict[int, str], int]:
    model_path = resolve_input_path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型目录: {model_path}")

    with (model_path / "train_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (model_path / "label2id.json").open("r", encoding="utf-8") as f:
        raw_label2id = json.load(f)
    label2id = {str(k): int(v) for k, v in raw_label2id.items()}
    id2label = {v: k for k, v in label2id.items()}

    bert_dir = model_path / "bert"
    bert_source = str(bert_dir) if bert_dir.exists() else config["bert_model_name_or_path"]
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    model = BertBiLSTMCRF(
        bert_model_name_or_path=bert_source,
        num_labels=len(label2id),
        lstm_hidden_size=int(config.get("lstm_hidden_size", 256)),
        lstm_layers=int(config.get("lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.1)),
    )
    device = device or resolve_device("auto")
    state_dict = torch.load(model_path / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    max_length = int(config.get("max_length", 256))
    return model, tokenizer, label2id, id2label, max_length


@torch.no_grad()
def predict_sentence(
    sentence: str,
    model: BertBiLSTMCRF,
    tokenizer: BertTokenizerFast,
    id2label: dict[int, str],
    max_length: int,
    device: torch.device | None = None,
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    device = device or resolve_device("auto")
    tokens = [tok for tok in jieba.lcut(sentence) if tok.strip()]
    if not tokens:
        return [], []

    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    if word_ids is None:
        return [], []

    crf_mask = torch.tensor(
        [[word_idx is not None for word_idx in word_ids]],
        dtype=torch.bool,
        device=device,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    _, predictions = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        crf_mask=crf_mask,
        labels=None,
        decode=True,
    )
    assert predictions is not None
    subtoken_pred = predictions[0]

    word_level_labels: list[str] = []
    pred_idx = 0
    prev_word_idx: int | None = None
    for word_idx in word_ids:
        if word_idx is None:
            continue
        pred_label = id2label[subtoken_pred[pred_idx]]
        pred_idx += 1
        if word_idx != prev_word_idx:
            word_level_labels.append(pred_label)
            prev_word_idx = word_idx

    token_label_pairs = list(zip(tokens[: len(word_level_labels)], word_level_labels))
    entities = decode_entities(token_label_pairs)
    return token_label_pairs, entities


def decode_entities(token_label_pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    current_type = ""
    current_tokens: list[str] = []

    def flush_current() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities.append({"type": current_type, "text": "".join(current_tokens)})
        current_type = ""
        current_tokens = []

    for token, label in token_label_pairs:
        if label.startswith("B-"):
            flush_current()
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(token)
        else:
            flush_current()

    flush_current()
    return entities


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BERT-BiLSTM-CRF 中文 NER 训练与推理脚本"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--train-file",
        default="data/processed/train.txt",
        help="训练集路径（相对路径按项目根目录解析）",
    )
    train_parser.add_argument("--bert-model", default="bert-base-chinese")
    train_parser.add_argument(
        "--output-dir",
        default="checkpoints/bert_bilstm_crf",
        help="模型保存目录（相对路径按项目根目录解析）",
    )
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=None,
        help="可选：BERT 之外层（BiLSTM+CRF）学习率，不设置则与 --learning-rate 相同",
    )
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--valid-ratio", type=float, default=0.1)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--lstm-hidden-size", type=int, default=256)
    train_parser.add_argument("--lstm-layers", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-steps", type=int, default=20, help="每多少个 batch 打印一次进度")
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（增大等效 batch）",
    )
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="早停耐心值，0 表示关闭早停",
    )
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="启用梯度检查点以减少显存占用",
    )
    train_parser.add_argument(
        "--mps-empty-cache-steps",
        type=int,
        default=0,
        help="MPS 训练时每 N step 执行一次 empty_cache，0 表示关闭",
    )
    train_parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="训练设备，auto 优先 MPS",
    )

    predict_parser = subparsers.add_parser("predict", help="单句预测")
    predict_parser.add_argument(
        "--model-dir",
        default="checkpoints/bert_bilstm_crf",
        help="模型目录（相对路径按项目根目录解析）",
    )
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="推理设备，auto 优先 MPS",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command in (None, "train"):
        if args.command is None:
            train()
            return
        train(
            train_file=args.train_file,
            bert_model_name_or_path=args.bert_model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            valid_ratio=args.valid_ratio,
            warmup_ratio=args.warmup_ratio,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            seed=args.seed,
            device_preference=args.device,
            log_steps=args.log_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stop_patience=args.early_stop_patience,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            mps_empty_cache_steps=args.mps_empty_cache_steps,
        )
        return

    if args.command == "predict":
        device = resolve_device(args.device)
        model, tokenizer, _label2id, id2label, max_length = load_model_for_inference(
            model_dir=args.model_dir,
            device=device,
        )
        token_label_pairs, entities = predict_sentence(
            sentence=args.text,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            max_length=max_length,
            device=device,
        )
        print("Token 标签:")
        for token, label in token_label_pairs:
            print(f"{token}\t{label}")
        print("实体结果:")
        for entity in entities:
            print(f"{entity['type']}\t{entity['text']}")


if __name__ == "__main__":
    main()
