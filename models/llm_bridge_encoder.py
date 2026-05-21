"""
C2: LLM-KG Bridge Encoder
Projects LLaMA-3-8B embeddings (4096-dim) into temporal KG space (256-dim)
via a two-layer MLP trained with MSE alignment loss on anchor entities.

Paper Section 3.2:
  h_PROJ = W2 σ(W1 h_LLM + b1) + b2
  ℓ_align = ||h_PROJ - e_v||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMBridgeEncoder(nn.Module):
    """
    LLM-KG Bridge Encoder (C2).

    Two-layer MLP that projects LLM embeddings into the KG embedding space.
    Trained via MSE loss over anchor entities (top-10% by KG degree centrality).

    Args:
        llm_dim (int): LLM embedding dimension (4096 for LLaMA-3-8B)
        kg_dim (int): KG embedding target dimension d (256)
        hidden_dim (int): MLP hidden dimension (default: 4*kg_dim = 1024)
        dropout (float): Dropout rate (default: 0.1)
    """

    def __init__(self, llm_dim=4096, kg_dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.llm_dim = llm_dim
        self.kg_dim = kg_dim

        # Two-layer MLP: h_PROJ = W2 σ(W1 h_LLM + b1) + b2
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, kg_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, llm_embeddings):
        """
        Args:
            llm_embeddings: [num_items, llm_dim] — pre-computed LLM embeddings

        Returns:
            projected: [num_items, kg_dim] — projected embeddings in KG space
        """
        return self.mlp(llm_embeddings)

    def alignment_loss(self, projected_anchor, kg_anchor):
        """
        MSE alignment loss over anchor entities.

        Args:
            projected_anchor: [num_anchors, kg_dim] — projected LLM embeddings
            kg_anchor: [num_anchors, kg_dim] — trained KG entity embeddings

        Returns:
            loss: scalar MSE loss ℓ_align = ||h_PROJ - e_v||²
        """
        return F.mse_loss(projected_anchor, kg_anchor.detach())


def select_anchors(kg_edge_index, num_entities, top_ratio=0.10, min_degree=3):
    """
    Select anchor entities as top-10% by KG degree centrality.
    Entities with fewer than min_degree=3 relations are excluded.

    Args:
        kg_edge_index: [2, num_edges] — KG edges in the most recent snapshot
        num_entities (int): Total number of entities
        top_ratio (float): Fraction of entities to use as anchors (default: 0.10)
        min_degree (int): Minimum KG degree to be eligible (default: 3)

    Returns:
        anchor_ids: [num_anchors] — entity indices selected as anchors
    """
    src = kg_edge_index[0]
    degree = torch.zeros(num_entities, dtype=torch.long)
    degree.scatter_add_(0, src, torch.ones_like(src))

    # Filter by minimum degree
    eligible = (degree >= min_degree).nonzero(as_tuple=True)[0]

    # Sort by degree and take top ratio
    top_k = max(1, int(len(eligible) * top_ratio))
    sorted_idx = eligible[degree[eligible].argsort(descending=True)]
    anchor_ids = sorted_idx[:top_k]

    return anchor_ids


def precompute_llm_embeddings(item_texts, model_name='meta-llama/Meta-Llama-3-8B',
                               batch_size=64, device='cuda', output_path=None):
    """
    Offline pre-computation of LLM embeddings for all items.
    Run ONCE before training; outputs cached to disk.

    Args:
        item_texts: list of str — item text descriptions
        model_name: HuggingFace model identifier
        batch_size: items per forward pass
        device: computation device
        output_path: path to save embeddings (.pt file)

    Returns:
        embeddings: [num_items, 4096] tensor
    """
    from transformers import AutoTokenizer, AutoModel
    import os

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,
                                      torch_dtype=torch.float16,
                                      device_map='auto')
    model.eval()

    all_embeddings = []
    for i in range(0, len(item_texts), batch_size):
        batch_texts = item_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt',
                           padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over token dimension
            emb = outputs.last_hidden_state.mean(dim=1)   # [B, 4096]
            all_embeddings.append(emb.cpu().float())

    embeddings = torch.cat(all_embeddings, dim=0)           # [N, 4096]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(embeddings, output_path)
        print(f"Saved LLM embeddings: {embeddings.shape} → {output_path}")

    return embeddings
