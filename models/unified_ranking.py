"""
C5: Unified Ranking Layer
Learnable fusion of GNN, LLM, and temporal affinity scores.

Paper Section 3.5:
  s(u,i) = β1·s_GNN(u,i) + β2·s_LLM(u,i) + β3·s_Temp(u,i)
  initialized: (β1, β2, β3) = (0.5, 0.3, 0.2)
"""

import torch
import torch.nn as nn


class UnifiedRankingLayer(nn.Module):
    """
    Unified Ranking Layer (C5).

    Fuses three affinity signals via jointly learned scalar weights.
    Weights are NOT constrained to sum to 1 — they adapt freely during training
    to reflect each dataset's optimal balance of temporal, semantic, structural signals.

    Observed convergence per dataset (paper Table 4):
        MIND          → β3 ≈ 0.45  (temporal-dominant: fast news cycles)
        FB15k-237     → β1 ≈ 0.60  (GNN-dominant: KG link-prediction task)
        Amazon Elec.  → β1≈0.35, β2≈0.38, β3≈0.27  (balanced)
        MovieLens-20M → β1≈0.42, β2≈0.32, β3≈0.26  (slight GNN dominance)

    Args:
        embedding_dim (int): Entity embedding dimension d (256)
        beta_init (tuple): Initial values for (β1, β2, β3) (default: (0.5, 0.3, 0.2))
    """

    def __init__(self, embedding_dim=256, beta_init=(0.5, 0.3, 0.2)):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Learnable fusion weights (unconstrained — no softmax)
        self.beta = nn.Parameter(torch.tensor(list(beta_init), dtype=torch.float32))

    def forward(self, user_emb, gnn_item_emb, llm_item_emb, temporal_item_emb):
        """
        Compute fused affinity score for (user, item) pairs.

        s(u,i) = β1·(e_u · e_i_gnn) + β2·(e_u · h_i_proj) + β3·(e_u · e_i_temp)

        Args:
            user_emb:          [B, d] — user embedding
            gnn_item_emb:      [B, d] — item embedding from C3 (GNN)
            llm_item_emb:      [B, d] — projected LLM item embedding from C2
            temporal_item_emb: [B, d] — temporal item embedding from C1

        Returns:
            score: [B] — final affinity scores
        """
        s_gnn  = (user_emb * gnn_item_emb).sum(dim=-1)       # [B]
        s_llm  = (user_emb * llm_item_emb).sum(dim=-1)        # [B]
        s_temp = (user_emb * temporal_item_emb).sum(dim=-1)   # [B]

        score = self.beta[0] * s_gnn + self.beta[1] * s_llm + self.beta[2] * s_temp
        return score                                            # [B]

    def get_fusion_weights(self):
        """Return current fusion weights (for logging/analysis)."""
        b = self.beta.detach().cpu()
        return {'beta_gnn': b[0].item(),
                'beta_llm': b[1].item(),
                'beta_temp': b[2].item()}
