# 红楼梦 NER (BERT-BiLSTM-CRF)

## 目录结构

```text
biye_sheji/
├── checkpoints/                  # 模型输出目录
├── data/
│   ├── raw/
│   │   └── honglou.txt
│   └── processed/
│       └── train.txt
├── src/
│   ├── build_pseudo_ner_data.py  # 伪标注数据生成
│   ├── train_bert_bilstm_crf.py  # 训练 + 推理
│   ├── inference.py              # 全量小说推理 + 共现三元组 + Neo4j
│   ├── qa_sbert.py               # 问答阶段 SBERT 语义关系匹配 + Cypher 生成
│   └── web_app.py                # 前端 API 服务
├── web/
│   ├── index.html                # 前端页面
│   ├── styles.css                # 页面样式
│   └── app.js                    # 前端交互逻辑
└── requirements.txt
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

先检查 MPS 是否可用：

```bash
python -c "import torch; print('mps_available=', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

## 2. 生成/重生伪标注数据

默认读取 `data/raw/honglou.txt`，输出到 `data/processed/train.txt`，支持人名和地名规则词表：

```bash
python src/build_pseudo_ner_data.py
```

可选参数：

```bash
python src/build_pseudo_ner_data.py \
  --per-file your_person_lexicon.txt \
  --loc-file your_location_lexicon.txt
```

## 3. 训练 BERT-BiLSTM-CRF

```bash
python src/train_bert_bilstm_crf.py train \
  --train-file data/processed/train.txt \
  --bert-model bert-base-chinese \
  --output-dir checkpoints/bert_bilstm_crf \
  --device mps \
  --epochs 3 \
  --batch-size 16
```

效果优先（推荐）：

```bash
python src/train_bert_bilstm_crf.py train \
  --train-file data/processed/train.txt \
  --bert-model bert-base-chinese \
  --output-dir checkpoints/bert_bilstm_crf \
  --device mps \
  --epochs 8 \
  --batch-size 8 \
  --max-length 256 \
  --learning-rate 2e-5 \
  --head-learning-rate 6e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --mps-empty-cache-steps 50 \
  --early-stop-patience 3 \
  --log-steps 20
```

训练完成后会保存：
- `checkpoints/bert_bilstm_crf/best_model.pt`
- `checkpoints/bert_bilstm_crf/label2id.json`
- `checkpoints/bert_bilstm_crf/train_config.json`
- `checkpoints/bert_bilstm_crf/bert/` (BERT 权重)

## 4. 对未见句子推理

```bash
python src/train_bert_bilstm_crf.py predict \
  --model-dir checkpoints/bert_bilstm_crf \
  --device mps \
  --text "贾宝玉在大观园遇见林黛玉。"
```

## 5. 全量抽取共现关系并写入 Neo4j

先自动抽取并生成两类三元组（不写库）：

```bash
python src/inference.py \
  --input data/raw/honglou.txt \
  --model-dir checkpoints/bert_bilstm_crf \
  --triples-path triples.csv \
  --relation-triples-path relation_triples.csv \
  --device mps \
  --skip-neo4j
```

写入 Neo4j（默认 `bolt://localhost:7687`）：

```bash
python src/inference.py \
  --input data/raw/honglou.txt \
  --model-dir checkpoints/bert_bilstm_crf \
  --triples-path triples.csv \
  --relation-triples-path relation_triples.csv \
  --device mps \
  --clear-existing \
  --neo4j-user neo4j \
  --neo4j-password 你的密码
```

说明：
- `triples.csv`：共现关系（`relation=共现`）。
- `relation_triples.csv`：自动抽取的语义关系（`父亲/母亲/住处/丫鬟`）。
- Neo4j 中关系包括：
  - `CO_OCCUR`（共现）
  - `RELATION`（语义关系，属性 `relation` 为父亲/母亲/住处/丫鬟）

## 6. 如何判断效果

- 训练日志中看 `F1`：每个 epoch 都会输出 `P/R/F1`，`F1` 越高通常越好。
- 先做人工 spot check：用第 4 步单句推理，看是否正确识别人名。
- 看关系抽取质量：打开 `triples.csv`，检查高频人物对是否合理。
- 使用 MPS 时 CPU 占用偏低是正常现象，主要计算在 GPU 完成；请在系统活动监视器里看 GPU 压力。

## 7. 问答阶段（SBERT 语义匹配）

脚本 `src/qa_sbert.py` 会做三件事：
- 用训练好的 NER 模型从问题里识别人名；
- 用 SBERT 计算问题与标准关系（如 `父亲,母亲,住处,丫鬟`）的余弦相似度；
- 选分数最高的关系生成 Cypher 查询语句，并仅通过 Neo4j 返回实体答案（不再使用规则兜底）。

示例：

```bash
python src/qa_sbert.py \
  --question "宝玉的父亲是哪个？" \
  --relations "父亲,母亲,住处,丫鬟" \
  --run-neo4j \
  --sbert-model BAAI/bge-small-zh-v1.5 \
  --ner-model-dir checkpoints/bert_bilstm_crf \
  --device mps \
  --neo4j-user neo4j \
  --neo4j-password 你的密码
```

如果要直接查询 Neo4j：

```bash
python src/qa_sbert.py \
  --question "宝玉的父亲是哪个？" \
  --relations "父亲,母亲,住处,丫鬟" \
  --run-neo4j \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password 你的密码
```

## 8. 前端控制台（推荐）

先完成第 5 步（自动关系抽取并写入 Neo4j），再启动前端问答。

安装新增依赖后启动：

```bash
python src/web_app.py --host 127.0.0.1 --port 8000
```

浏览器打开：
- [http://127.0.0.1:8000](http://127.0.0.1:8000)

页面能力：
- `NER 调试`：输入句子，直接看到 token 的 BIO 标签和实体识别结果。
- `问答查询`：输入问题，自动展示最终答案、人名识别、关系相似度排名、Cypher、可选 Neo4j 查询结果。

## 注意

若你的 `train.txt` 没有 `B-LOC/I-LOC` 样本，模型无法真正学会地名识别。  
建议补充地名词表后先重生 `train.txt` 再训练。
