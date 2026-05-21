"""
C3: Dynamic Graph Propagation
L-layer time-decay message passing over temporal KG snapshot + InfoNCE contrastive SSL.

Paper Section 3.3:
  e_v(l)(t) = σ( Wl · ( e_v(l-1)(t) + Σ_{n∈N(v)} α(t,t'n) · e_n(l-1)(t) ) )
  ℓ_ssl = -Σ_v log( exp(sim(e_v^A, e_v^B)/τ) / Σ_{u≠v} exp(sim(e_v^A, e_u^B)/τ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDecayGNNLayer(nn.Module):
    """Single GNN layer with time-decay weighted neighborhood aggregation."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_decay_weights):
        """
        Args:
            x: [N, d] — entity embeddings at layer l-1
            edge_index: [2, E] — (src, dst) entity indices
            edge_decay_weights: [E] — α(t, t') time-decay weights

        Returns:
            x_new: [N, d] — updated embeddings at layer l
        """
        src, dst = edge_index[0], edge_index[1]
        N, d = x.shape

        # Weighted neighbor aggregation
        neighbor_emb = x[dst]                                  # [E, d]
        weighted = edge_decay_weights.unsqueeze(1) * neighbor_emb  # [E, d]

        agg = torch.zeros(N, d, device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, d), weighted)
        count.scatter_add_(0, src.unsqueeze(1), torch.ones(len(src), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count                                       # mean aggregation

        # Layer update: e_v(l) = σ(W_l · (e_v(l-1) + agg))
        x_new = self.activation(self.W(x + agg))
        return x_new


class DynamicGraphPropagation(nn.Module):
    """
    Dynamic Graph Propagation (C3).

    L-layer time-decay GNN over the most recent temporal KG snapshot,
    with InfoNCE contrastive SSL to prevent over-smoothing.

    Args:
        embedding_dim (int): Entity embedding dimension d (default: 256)
        num_layers (int): Number of GNN propagation layers L (default: 3)
        ssl_temperature (float): InfoNCE temperature τ (default: 0.2)
        edge_dropout (float): Fraction of edges dropped for View A (default: 0.20)
        feature_mask_ratio (float): Fraction of features masked for View B (default: 0.15)
        lambda_decay (float): Time-decay coefficient λ (default: 0.1)
    """

    def __init__(self, embedding_dim=256, num_layers=3, ssl_temperature=0.2,
                 edge_dropout=0.20, feature_mask_ratio=0.15, lambda_decay=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.ssl_temperature = ssl_temperature
        self.edge_dropout = edge_dropout
        self.feature_mask_ratio = feature_mask_ratio
        self.lambda_decay = lambda_decay

        # L independent GNN layers
        self.layers = nn.ModuleList([
            TimeDecayGNNLayer(embedding_dim) for _ in range(num_layers)
        ])

    def compute_time_decay(self, t_current, edge_timestamps):
        """α(t, t') = exp(-λ(t - t'))"""
        return torch.exp(-self.lambda_decay * (t_current - edge_timestamps).float())

    def forward(self, x, edge_index, edge_attr, t_current=None):
        """
        Args:
            x: [N, d] — entity embeddings from C1
            edge_index: [2, E] — graph edges
            edge_attr: [E] — edge timestamps
            t_current: float — current snapshot time (days from epoch)

        Returns:
            entity_emb: [N, d] — final entity embeddings (mean-pooled over L layers)
            ssl_loss: scalar — InfoNCE contrastive loss
        """
        # Compute time-decay weights
        if t_current is None:
            t_current = edge_attr.max().item()
        alpha = self.compute_time_decay(t_current, edge_attr)  # [E]

        # L-layer propagation with mean-pooling
        layer_outputs = []
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, alpha)
            layer_outputs.append(h)

        # Mean-pool across all L layers
        entity_emb = torch.stack(layer_outputs, dim=0).mean(dim=0)  # [N, d]

        # Contrastive SSL loss
        ssl_loss = self._contrastive_ssl(x, edge_index, alpha)

        return entity_emb, ssl_loss

    def _augment_view_A(self, edge_index, alpha):
        """View A: randomly drop 20% of edges."""
        mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.edge_dropout
        return edge_index[:, mask], alpha[mask]

    def _augment_view_B(self, x):
        """View B: randomly mask 15% of entity feature dimensions."""
        mask = torch.rand_like(x) > self.feature_mask_ratio
        return x * mask

    def _encode_view(self, x, edge_index, alpha):
        """Encode a single augmented view through all L layers (mean-pooled)."""
        layer_out = []
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, alpha)
            layer_out.append(h)
        return torch.stack(layer_out, dim=0).mean(dim=0)

    def _contrastive_ssl(self, x, edge_index, alpha):
        """
        InfoNCE contrastive loss.

        ℓ_ssl = -Σ_v log(exp(sim(e_v^A, e_v^B)/τ) / Σ_{u≠v} exp(sim(e_v^A, e_u^B)/τ))
        """
        # Build augmented views
        ei_A, alpha_A = self._augment_view_A(edge_index, alpha)
        x_B = self._augment_view_B(x)

        # Encode both views
        z_A = F.normalize(self._encode_view(x, ei_A, alpha_A), dim=1)    # [N, d]
        z_B = F.normalize(self._encode_view(x_B, edge_index, alpha), dim=1)  # [N, d]

        # Similarity matrix: [N, N]
        sim = torch.matmul(z_A, z_B.T) / self.ssl_temperature

        # InfoNCE: diagonal is positive pair, all others are negatives
        labels = torch.arange(z_A.size(0), device=z_A.device)
        loss = F.cross_entropy(sim, labels)
        return loss
