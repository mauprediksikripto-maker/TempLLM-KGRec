"""
C1: Temporal KG Builder
Builds ordered KG snapshots and encodes entity temporal evolution via GRU.

Paper Section 3.1:
  e_v(t) = GRU(e_v(t-1), Δe_v(t))
  α(t, t') = exp(-λ(t - t'))
  x_v(t) = (1/|N_t(v)|) * Σ_{u∈N_t(v)} α(t, t_uv) * e_u(t-1)
"""

import torch
import torch.nn as nn
import math


class TemporalKGBuilder(nn.Module):
    """
    Temporal KG Builder (C1).

    Slices interaction history into non-overlapping monthly snapshots and encodes
    entity embeddings via a GRU with time-decay neighborhood aggregation.

    Args:
        entity_dim (int): KG embedding dimension d (default: 256)
        gru_hidden_dim (int): GRU hidden state dimension d_h (default: 256)
        lambda_decay (float): Time-decay coefficient λ (default: 0.1)
        delta_t (int): Snapshot window in days (default: 30)
        min_lifetime_days (int): Entities active fewer days are discarded (default: 5)
    """

    def __init__(self, entity_dim=256, gru_hidden_dim=256,
                 lambda_decay=0.1, delta_t=30, min_lifetime_days=5):
        super().__init__()
        self.entity_dim = entity_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.lambda_decay = lambda_decay
        self.delta_t = delta_t
        self.min_lifetime_days = min_lifetime_days

        # GRU encoder: propagates entity state forward through snapshots
        self.gru = nn.GRUCell(input_size=entity_dim + 1,  # +1 for lifetime scalar
                               hidden_size=gru_hidden_dim)

        # Projection: GRU hidden state → KG embedding space
        self.proj = nn.Linear(gru_hidden_dim, entity_dim)

        # Cold-entity bias for entities in fewer than 2 snapshots
        self.cold_entity_bias = nn.Parameter(torch.zeros(entity_dim))

    def time_decay(self, t_current, t_edge):
        """α(t, t') = exp(-λ(t - t'))"""
        return torch.exp(-self.lambda_decay * (t_current - t_edge).float())

    def forward(self, snapshots):
        """
        Args:
            snapshots: list of B dicts, each containing:
                - edge_index: [2, num_edges] — source and target entity indices
                - edge_attr: [num_edges] — edge timestamps (in days from epoch)
                - entity_ids: [num_entities] — entity indices in this snapshot
                - entity_lifetimes: [num_entities] — days active at snapshot time
                - snapshot_time: float — timestamp of this snapshot (days)

        Returns:
            entity_emb: [num_entities_total, entity_dim] — final temporal embeddings
        """
        num_entities = snapshots[0]['entity_ids'].max().item() + 1
        device = snapshots[0]['edge_index'].device

        # Initialize hidden states to zero
        h = torch.zeros(num_entities, self.gru_hidden_dim, device=device)
        snapshot_count = torch.zeros(num_entities, device=device)

        for snap in snapshots:
            t = snap['snapshot_time']
            edge_index = snap['edge_index']       # [2, E]
            edge_times = snap['edge_attr']        # [E] timestamps

            # Time-decay weights for each edge
            alpha = self.time_decay(t, edge_times)  # [E]

            # Aggregate time-decayed neighbor embeddings for each entity
            src, dst = edge_index[0], edge_index[1]
            neighbor_emb = self.proj(h)[dst]         # [E, d]
            weighted = alpha.unsqueeze(1) * neighbor_emb  # [E, d]

            # Mean aggregation per entity: x_v(t)
            x = torch.zeros(num_entities, self.entity_dim, device=device)
            count = torch.zeros(num_entities, 1, device=device)
            x.scatter_add_(0, src.unsqueeze(1).expand_as(weighted), weighted)
            count.scatter_add_(0, src.unsqueeze(1),
                                torch.ones(len(src), 1, device=device))
            count = count.clamp(min=1)
            x = x / count  # normalize

            # Append entity lifetime scalar feature
            lifetimes = snap['entity_lifetimes'].float().unsqueeze(1)  # [N, 1]
            x_with_lifetime = torch.cat([x, lifetimes], dim=1)         # [N, d+1]

            # GRU update
            h_new = self.gru(x_with_lifetime, h)

            # Only update entities present in this snapshot
            entity_mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
            entity_mask[snap['entity_ids']] = True
            h = torch.where(entity_mask.unsqueeze(1), h_new, h)
            snapshot_count[snap['entity_ids']] += 1

        # Project to KG embedding space
        entity_emb = self.proj(h)

        # Apply cold-entity bias for entities with < 2 snapshot appearances
        cold_mask = (snapshot_count < 2).unsqueeze(1)
        entity_emb = entity_emb + cold_mask.float() * self.cold_entity_bias.unsqueeze(0)

        return entity_emb  # [num_entities, entity_dim]
