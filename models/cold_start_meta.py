"""
C4: Cold-Start Meta-Bridge
Schema-conditioned MAML for fast adaptation with ≤5 user interactions.

Paper Section 3.4:
  θ_i' = θ - α_i ∇_θ ℓ_BPR(S_i; θ)
  ℓ_MAML = Σ_i ℓ_BPR(Q_i; θ_i')
  p(u) = MLP(ϕ(T_KG))  — schema-conditioned initialization bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class KGSchemaEncoder(nn.Module):
    """
    Encodes temporal KG schema statistics into a bias vector p(u).

    Schema statistics for a user's support set items:
        - Entity type distribution (normalized histogram)
        - Relation type frequency (normalized histogram)
        - Mean entity lifetime (scalar, normalized)
        - Mean entity degree in latest snapshot (scalar, normalized)
    """

    def __init__(self, schema_dim, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(schema_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, schema_stats):
        """
        Args:
            schema_stats: [B, schema_dim] — concatenated KG schema statistics

        Returns:
            p: [B, embedding_dim] — schema-conditioned initialization bias
        """
        return self.mlp(schema_stats)


class ColdStartMetaBridge(nn.Module):
    """
    Cold-Start Meta-Bridge (C4).

    Wraps MAML around the full model with a schema-conditioned initialization
    that uses temporal KG structure as a task-agnostic prior.

    Args:
        embedding_dim (int): Entity/user embedding dimension (256)
        schema_dim (int): Dimension of KG schema statistics vector
        inner_lr (float): MAML inner-loop learning rate α_i (default: 0.01)
        support_size (int): Number of support interactions K (default: 5)
        inner_steps (int): Number of inner-loop gradient steps (default: 2)
    """

    def __init__(self, embedding_dim=256, schema_dim=64,
                 inner_lr=0.01, support_size=5, inner_steps=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.inner_lr = inner_lr
        self.support_size = support_size
        self.inner_steps = inner_steps

        # Schema encoder: produces bias vector p(u) = MLP(ϕ(T_KG))
        self.schema_encoder = KGSchemaEncoder(schema_dim, embedding_dim)

        # Learnable gate for combining base init and schema bias
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)

    def _bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        """Bayesian Personalized Ranking loss."""
        pos_score = (user_emb * pos_item_emb).sum(dim=-1)
        neg_score = (user_emb * neg_item_emb).sum(dim=-1)
        return -F.logsigmoid(pos_score - neg_score).mean()

    def adapt(self, support_set, kg_schema_stats, base_user_emb):
        """
        Schema-conditioned MAML adaptation for a new user.

        Args:
            support_set: dict with {pos_item_emb, neg_item_emb} — K interactions
            kg_schema_stats: [schema_dim] — KG schema statistics of support items
            base_user_emb: [embedding_dim] — base user embedding from main model

        Returns:
            adapted_user_emb: [embedding_dim] — adapted embedding after inner loop
        """
        # Compute schema-conditioned initialization bias: p(u)
        schema_bias = self.schema_encoder(
            kg_schema_stats.unsqueeze(0)
        ).squeeze(0)                                            # [d]

        # Combine base init with schema bias via gating
        init = self.gate(
            torch.cat([base_user_emb, schema_bias], dim=-1)
        )                                                       # [d]

        # Inner-loop adaptation (2 gradient steps by default)
        adapted = init.clone().requires_grad_(True)
        inner_optimizer = torch.optim.SGD([adapted], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            loss = self._bpr_loss(
                adapted.unsqueeze(0).expand(self.support_size, -1),
                support_set['pos_item_emb'],
                support_set['neg_item_emb'],
            )
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted.detach()

    def meta_train_step(self, meta_tasks, model, outer_optimizer):
        """
        MAML outer-loop training step.

        ℓ_MAML = Σ_i ℓ_BPR(Q_i; θ_i')

        Args:
            meta_tasks: list of dicts, each with {support_set, query_set, kg_schema_stats}
            model: full TempLLMKGRec model (for accessing user embeddings)
            outer_optimizer: optimizer for outer-loop update

        Returns:
            meta_loss: scalar — sum of query BPR losses
        """
        outer_optimizer.zero_grad()
        total_meta_loss = 0.0

        for task in meta_tasks:
            # Inner loop: adapt to support set
            user_id = task['user_id']
            base_emb = model.user_embeddings(user_id)

            adapted_emb = self.adapt(
                support_set=task['support_set'],
                kg_schema_stats=task['kg_schema_stats'],
                base_user_emb=base_emb,
            )

            # Outer loop: evaluate on query set
            query_loss = self._bpr_loss(
                adapted_emb.unsqueeze(0).expand(len(task['query_set']['pos_item_emb']), -1),
                task['query_set']['pos_item_emb'],
                task['query_set']['neg_item_emb'],
            )
            total_meta_loss += query_loss

        total_meta_loss.backward()
        outer_optimizer.step()

        return total_meta_loss.item()


def extract_kg_schema_stats(item_ids, kg_snapshot, entity_types,
                             relation_types, entity_lifetimes, num_type_bins=32):
    """
    Extract temporal KG schema statistics for a set of items (support set).

    Statistics extracted:
        - Entity type distribution: normalized histogram over entity_types
        - Relation type frequencies: normalized histogram over relation_types in 1-hop neighborhood
        - Mean entity lifetime (normalized by max_lifetime)
        - Mean entity degree in latest snapshot (normalized by max_degree)

    Args:
        item_ids: list of int — item entity IDs in support set
        kg_snapshot: dict — latest KG snapshot (edge_index, edge_attr)
        entity_types: [num_entities] — entity type label per entity
        relation_types: [num_edges] — relation type label per edge
        entity_lifetimes: [num_entities] — lifetime in days
        num_type_bins (int): number of bins for type histograms (default: 32)

    Returns:
        schema_stats: [schema_dim] tensor — concatenated schema statistics
    """
    item_ids_t = torch.tensor(item_ids)

    # Entity type distribution
    types = entity_types[item_ids_t]
    type_hist = torch.zeros(num_type_bins)
    for t in types:
        type_hist[t % num_type_bins] += 1
    type_hist = type_hist / (type_hist.sum() + 1e-8)

    # Mean entity lifetime (normalized)
    lifetimes = entity_lifetimes[item_ids_t].float()
    mean_lifetime = lifetimes.mean() / (entity_lifetimes.max() + 1e-8)

    # Mean degree in latest snapshot
    src = kg_snapshot['edge_index'][0]
    degree = torch.zeros(entity_lifetimes.shape[0])
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    mean_degree = degree[item_ids_t].mean() / (degree.max() + 1e-8)

    # Concatenate: type_hist (32) + mean_lifetime (1) + mean_degree (1) = 34
    schema_stats = torch.cat([
        type_hist,
        mean_lifetime.unsqueeze(0),
        mean_degree.unsqueeze(0),
    ])

    return schema_stats
