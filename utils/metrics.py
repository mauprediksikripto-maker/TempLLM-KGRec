"""
Evaluation Metrics for TempLLM-KGRec
Full-ranking protocol: NDCG@K, Recall@K, Hit@K

Protocol (paper Section 4.3):
  - Full-ranking: score test item against ALL non-interacted catalog items
  - No sampled negative pool (eliminates sampling bias)
  - Primary: NDCG@10, Recall@10
  - Supplementary: NDCG@20, Recall@20 (consistent trends)
"""

import torch
import numpy as np
from scipy.stats import wilcoxon


def ndcg_at_k(ranked_items, pos_item, k=10):
    """
    Compute NDCG@K for a single user.

    Args:
        ranked_items: list — item IDs ranked by score (descending)
        pos_item: int — ground-truth positive test item
        k: int — cutoff (default: 10)

    Returns:
        ndcg: float
    """
    if pos_item in ranked_items[:k]:
        rank = ranked_items[:k].index(pos_item) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def recall_at_k(ranked_items, pos_item, k=10):
    """
    Compute Recall@K for a single user (binary: item in top-k or not).

    Args:
        ranked_items: list — ranked item IDs
        pos_item: int — ground-truth positive test item
        k: int — cutoff (default: 10)

    Returns:
        recall: float (0.0 or 1.0)
    """
    return 1.0 if pos_item in ranked_items[:k] else 0.0


def evaluate_model(model, test_data, kg_snapshots, llm_embeddings,
                   k_list=(10, 20), cold_start_k=5, device='cuda'):
    """
    Full-ranking evaluation over all test users.

    Args:
        model: TempLLMKGRec — trained model
        test_data: list of dicts — {user_id, pos_item_id, candidate_items, num_train_interactions}
        kg_snapshots: list of dicts — temporal KG snapshots
        llm_embeddings: [num_items, llm_dim] tensor — pre-computed LLM embeddings
        k_list: tuple — evaluation cutoffs (default: (10, 20))
        cold_start_k: int — threshold for cold-start regime (default: 5)
        device: str

    Returns:
        results: dict — {NDCG@K, Recall@K} for warm/cold-start splits
    """
    model.eval()
    metrics = {f'NDCG@{k}': [] for k in k_list}
    metrics.update({f'Recall@{k}': [] for k in k_list})
    cold_metrics = {f'Cold_NDCG@{k}': [] for k in k_list}
    cold_metrics.update({f'Cold_Recall@{k}': [] for k in k_list})

    with torch.no_grad():
        for user_data in test_data:
            user_id = user_data['user_id']
            pos_item = user_data['pos_item_id']
            candidate_items = user_data['candidate_items']
            num_train = user_data['num_train_interactions']

            # Score all candidates
            scores = model.get_all_item_scores(
                user_id=torch.tensor([user_id], device=device),
                kg_snapshots=kg_snapshots,
                llm_embeddings=llm_embeddings,
                item_ids=torch.tensor(candidate_items, device=device),
            )

            # Rank by score
            ranked_idx = scores.argsort(descending=True).cpu().tolist()
            ranked_items = [candidate_items[i] for i in ranked_idx]

            # Compute metrics
            is_cold = num_train <= cold_start_k

            for k in k_list:
                ndcg = ndcg_at_k(ranked_items, pos_item, k)
                rec = recall_at_k(ranked_items, pos_item, k)
                metrics[f'NDCG@{k}'].append(ndcg)
                metrics[f'Recall@{k}'].append(rec)
                if is_cold:
                    cold_metrics[f'Cold_NDCG@{k}'].append(ndcg)
                    cold_metrics[f'Cold_Recall@{k}'].append(rec)

    # Average metrics
    results = {}
    for key, values in {**metrics, **cold_metrics}.items():
        if values:
            results[key] = np.mean(values)

    return results


def wilcoxon_significance_test(scores_ours, scores_baseline, alpha=0.05):
    """
    Wilcoxon signed-rank test for statistical significance.

    Paper Section 5.1:
      "All reported differences are statistically significant under
       the Wilcoxon signed-rank test (p < 0.05) across five independent runs."

    Args:
        scores_ours: list of float — NDCG@10 across 5 runs (our method)
        scores_baseline: list of float — NDCG@10 across 5 runs (baseline)
        alpha: float — significance level (default: 0.05)

    Returns:
        is_significant: bool
        p_value: float
    """
    statistic, p_value = wilcoxon(scores_ours, scores_baseline)
    return p_value < alpha, p_value


def retention_rate(warm_ndcg, cold_ndcg):
    """
    Warm-to-cold retention rate (paper Section 5.2).
    Measures graceful degradation under interaction sparsity.

    Retention = Cold NDCG@10 / Warm NDCG@10 × 100%
    """
    return (cold_ndcg / warm_ndcg) * 100 if warm_ndcg > 0 else 0.0
