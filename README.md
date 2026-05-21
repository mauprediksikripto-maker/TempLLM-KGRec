# TempLLM-KGRec

**Temporal Knowledge Graph Reasoning with Large Language Model Integration for Cold-Start Recommendation**

> Dedi Irawan, Sudarmaji  
> Fakultas Ilmu Komputer, Universitas Muhammadiyah Metro, Indonesia  
> *International Journal of Advances in Intelligent Informatics (IJAIN), 2025*

---

## Overview

TempLLM-KGRec is an end-to-end recommendation framework that jointly addresses:
- **Temporal drift** — user preferences evolve over time
- **LLM-KG misalignment** — bridging large language model embeddings with temporal KG space
- **Cold-start fragility** — reliable recommendations with ≤5 user interactions

The framework comprises five jointly trained components:

| Component | Role |
|-----------|------|
| **C1** Temporal KG Builder | Monthly snapshots with time-decay attention (GRU encoder) |
| **C2** LLM-KG Bridge Encoder | Projects LLaMA-3-8B (4096-dim) → KG space (256-dim) via MLP |
| **C3** Dynamic Graph Propagation | L=3 layer message passing + contrastive SSL (InfoNCE) |
| **C4** Cold-Start Meta-Bridge | Schema-conditioned MAML with temporal KG prior |
| **C5** Unified Ranking Layer | Learnable fusion of GNN + LLM + temporal scores |

## Results

| Dataset | NDCG@10 | Recall@10 | Improvement vs best baseline |
|---------|---------|-----------|-------------------------------|
| MovieLens-20M | 0.4797 | 0.5991 | +8.5% |
| Amazon Electronics | 0.3347 | 0.4288 | +11.2% |
| FB15k-237 | 0.3842 | 0.4790 | +8.5% |
| MIND | 0.3677 | 0.4589 | +8.5% |

Cold-start (≤5 interactions): **+35.9%** over strongest baseline (KGLLMRec).

---

## Requirements

```
Python >= 3.9
PyTorch >= 2.1.0
transformers >= 4.35.0   # for LLaMA-3-8B
torch-geometric >= 2.4.0
faiss-gpu >= 1.7.4
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
tqdm >= 4.65.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **Full training** (as in paper): NVIDIA A100 80GB GPU
- **Reduced training** (batch_size=512): NVIDIA RTX 3090/4090 (24GB GPU)
- **Inference only**: Any GPU with ≥8GB VRAM

---

## Installation

```bash
git clone https://github.com/anonymous-submission/TempLLM-KGRec.git
cd TempLLM-KGRec
pip install -r requirements.txt
```

---

## Dataset Preparation

### Download Datasets

| Dataset | Source | License |
|---------|--------|---------|
| MovieLens-20M | [GroupLens](https://grouplens.org/datasets/movielens/20m/) | Custom (non-commercial) |
| Amazon Electronics | [UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) | Custom |
| FB15k-237 | [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=52312) | MIT |
| MIND | [Microsoft](https://msnews.github.io/) | Custom |

Place raw data in `data/raw/<dataset_name>/`.

### Preprocess

```bash
# MovieLens-20M
python scripts/preprocess.py --dataset movielens --min_rating 4 --min_interactions 5

# Amazon Electronics
python scripts/preprocess.py --dataset amazon_electronics --min_rating 4 --min_interactions 5

# FB15k-237
python scripts/preprocess.py --dataset fb15k237

# MIND
python scripts/preprocess.py --dataset mind --binarize_clicks True
```

### Build KG Snapshots (C1)

```bash
python scripts/build_snapshots.py \
    --dataset movielens \
    --delta_t 30 \
    --lambda_decay 0.1 \
    --min_lifetime 5
```

### Pre-compute LLM Embeddings (C2, offline step)

```bash
python scripts/precompute_llm.py \
    --dataset movielens \
    --model_name meta-llama/Meta-Llama-3-8B \
    --batch_size 64 \
    --output_dir data/processed/movielens/llm_embeddings/
```

> **Note:** This step requires ~3.2 hours on A100 for MIND dataset. The embeddings are cached and not re-computed during training.

---

## Training

### Full Training (A100 80GB)

```bash
python train.py \
    --dataset movielens \
    --config configs/movielens.yaml
```

### Reduced Memory Training (RTX 3090/4090, 24GB)

```bash
python train.py \
    --dataset movielens \
    --config configs/movielens.yaml \
    --batch_size 512 \
    --grad_accum_steps 4
```

### Configuration

All hyperparameters are in `configs/<dataset>.yaml`. Key parameters:

```yaml
# Model
embedding_dim: 256
gnn_layers: 3
llm_dim: 4096

# C1 - Temporal KG Builder
delta_t: 30          # days per snapshot
lambda_decay: 0.1    # time-decay coefficient

# C3 - Dynamic Graph Propagation
ssl_temperature: 0.2
edge_dropout: 0.20
feature_mask_ratio: 0.15

# C4 - Cold-Start Meta-Bridge
maml_inner_lr: 0.01
maml_support_size: 5
maml_inner_steps: 2

# C5 - Unified Ranking Layer
beta_init: [0.5, 0.3, 0.2]   # GNN, LLM, Temporal weights

# Training
learning_rate: 0.001
batch_size: 2048
negative_samples: 200
warmup_epochs: 30
patience: 10
max_epochs: 200

# Loss weights (grid searched over {0.01, 0.05, 0.1, 0.5})
mu1: 0.05    # SSL loss weight
mu2: 0.1     # Alignment loss weight
```

---

## Evaluation

### Full Evaluation (all baselines + all datasets)

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset movielens \
    --eval_protocol full_ranking \
    --cold_start_k 5
```

### Reproduce Paper Results

```bash
bash scripts/reproduce_paper.sh
```

This script runs all experiments from Tables 4, 5, and 6 in the paper.

---

## Ablation Study

```bash
# w/o C1 (static KG)
python train.py --config configs/movielens.yaml --ablation no_temporal

# w/o C2 (random LLM projection)
python train.py --config configs/movielens.yaml --ablation no_llm_align

# w/o C3 (mean aggregation, no time-decay)
python train.py --config configs/movielens.yaml --ablation no_dynamic_prop

# w/o C4 (random cold-start init)
python train.py --config configs/movielens.yaml --ablation no_meta_bridge

# w/o SSL (no contrastive loss)
python train.py --config configs/movielens.yaml --ablation no_ssl

# w/o Joint (sequential pre-training)
python train.py --config configs/movielens.yaml --ablation sequential
```

---

## Pre-trained Models

Pre-trained checkpoints for all four datasets are available at:

| Dataset | Checkpoint | NDCG@10 |
|---------|-----------|---------|
| MovieLens-20M | [download](checkpoints/) | 0.4797 |
| Amazon Electronics | [download](checkpoints/) | 0.3347 |
| FB15k-237 | [download](checkpoints/) | 0.3842 |
| MIND | [download](checkpoints/) | 0.3677 |

---

## Repository Structure

```
TempLLM-KGRec/
├── models/
│   ├── temporal_kg_builder.py     # C1: GRU-based temporal encoder
│   ├── llm_bridge_encoder.py      # C2: LLM-KG projection MLP
│   ├── dynamic_graph_prop.py      # C3: GNN + contrastive SSL
│   ├── cold_start_meta.py         # C4: MAML cold-start module
│   ├── unified_ranking.py         # C5: Fusion ranking layer
│   └── templlm_kgrec.py           # Full model integration
├── data/
│   ├── raw/                       # Downloaded raw datasets
│   └── processed/                 # Preprocessed snapshots + embeddings
├── configs/
│   ├── movielens.yaml
│   ├── amazon_electronics.yaml
│   ├── fb15k237.yaml
│   └── mind.yaml
├── scripts/
│   ├── preprocess.py
│   ├── build_snapshots.py
│   ├── precompute_llm.py
│   └── reproduce_paper.sh
├── experiments/
│   └── hyperparameter_logs/       # Full grid-search logs (Table 4 paper)
├── utils/
│   ├── metrics.py                 # NDCG@K, Recall@K, full-ranking eval
│   ├── negative_sampling.py
│   └── temporal_utils.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@article{irawan2025templlmkgrec,
  title={TempLLM-KGRec: Temporal Knowledge Graph Reasoning with Large Language Model Integration for Cold-Start Recommendation},
  author={Irawan, Dedi and Sudarmaji},
  journal={International Journal of Advances in Intelligent Informatics},
  volume={11},
  year={2025},
  publisher={Universitas Ahmad Dahlan},
  doi={10.26555/ijain.v11iX.XXXX}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
