"""
TempLLM-KGRec: Full Model Integration
Temporal Knowledge Graph Reasoning with LLM Integration for Cold-Start Recommendation

Paper: Irawan & Sudarmaji, IJAIN 2025
"""

import torch
import torch.nn as nn
from models.temporal_kg_builder import TemporalKGBuilder
from models.llm_bridge_encoder import LLMBridgeEncoder
from models.dynamic_graph_prop import DynamicGraphPropagation
from models.cold_start_meta import ColdStartMetaBridge
from models.unified_ranking import UnifiedRankingLayer


class TempLLMKGRec(nn.Module):
    """
    End-to-end recommendation framework with five jointly optimized components.

    Components:
        C1: Temporal KG Builder     - GRU-based temporal entity encoder
        C2: LLM-KG Bridge Encoder   - Projects LLaMA-3-8B embeddings → KG space
        C3: Dynamic Graph Prop.     - L-layer GNN with time-decay + contrastive SSL
        C4: Cold-Start Meta-Bridge  - Schema-conditioned MAML
        C5: Unified Ranking Layer   - Learnable fusion of GNN + LLM + temporal scores
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # C1: Temporal KG Builder
        self.temporal_builder = TemporalKGBuilder(
            entity_dim=config.embedding_dim,
            gru_hidden_dim=config.embedding_dim,
            lambda_decay=config.lambda_decay,
            delta_t=config.delta_t,
        )

        # C2: LLM-KG Bridge Encoder
        self.llm_bridge = LLMBridgeEncoder(
            llm_dim=config.llm_dim,         # 4096 for LLaMA-3-8B
            kg_dim=config.embedding_dim,    # 256
            hidden_dim=config.embedding_dim * 4,
        )

        # C3: Dynamic Graph Propagation
        self.graph_prop = DynamicGraphPropagation(
            embedding_dim=config.embedding_dim,
            num_layers=config.gnn_layers,       # L = 3
            ssl_temperature=config.ssl_temperature,  # τ = 0.2
            edge_dropout=config.edge_dropout,        # 0.20
            feature_mask_ratio=config.feature_mask_ratio,  # 0.15
        )

        # C4: Cold-Start Meta-Bridge
        self.meta_bridge = ColdStartMetaBridge(
            embedding_dim=config.embedding_dim,
            inner_lr=config.maml_inner_lr,      # 0.01
            support_size=config.maml_support_size,  # K = 5
            inner_steps=config.maml_inner_steps,    # 2
        )

        # C5: Unified Ranking Layer
        self.ranking_layer = UnifiedRankingLayer(
            embedding_dim=config.embedding_dim,
            beta_init=config.beta_init,  # (0.5, 0.3, 0.2)
        )

        # User embedding table
        self.user_embeddings = nn.Embedding(config.num_users, config.embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)

    def forward(self, batch, kg_snapshots, llm_embeddings, mode='train'):
        """
        Forward pass.

        Args:
            batch: dict with keys {user_ids, pos_item_ids, neg_item_ids}
            kg_snapshots: list of B temporal KG snapshots (edge_index, edge_attr, timestamps)
            llm_embeddings: pre-computed LLM embeddings [num_items, llm_dim]
            mode: 'train' | 'eval' | 'cold_start'

        Returns:
            loss (train mode) or scores (eval mode)
        """
        # C1: Build temporal entity embeddings from KG snapshots
        temporal_entity_emb = self.temporal_builder(kg_snapshots)  # [num_entities, d]

        # C2: Project LLM embeddings into KG space (offline cached, no gradient)
        with torch.no_grad():
            projected_llm_emb = self.llm_bridge(llm_embeddings)    # [num_items, d]

        # C3: Dynamic graph propagation over latest snapshot + contrastive SSL
        gnn_entity_emb, ssl_loss = self.graph_prop(
            x=temporal_entity_emb,
            edge_index=kg_snapshots[-1]['edge_index'],
            edge_attr=kg_snapshots[-1]['edge_attr'],
        )                                                           # [num_entities, d]

        # C5: Compute final affinity scores
        user_emb = self.user_embeddings(batch['user_ids'])          # [B, d]
        pos_item_emb = gnn_entity_emb[batch['pos_item_ids']]        # [B, d]
        neg_item_emb = gnn_entity_emb[batch['neg_item_ids']]        # [B, d]

        pos_llm_emb = projected_llm_emb[batch['pos_item_ids']]      # [B, d]
        neg_llm_emb = projected_llm_emb[batch['neg_item_ids']]      # [B, d]

        pos_temp_emb = temporal_entity_emb[batch['pos_item_ids']]   # [B, d]
        neg_temp_emb = temporal_entity_emb[batch['neg_item_ids']]   # [B, d]

        pos_score = self.ranking_layer(user_emb, pos_item_emb, pos_llm_emb, pos_temp_emb)
        neg_score = self.ranking_layer(user_emb, neg_item_emb, neg_llm_emb, neg_temp_emb)

        if mode == 'train':
            # BPR loss
            bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

            # MSE alignment loss (C2)
            align_loss = self.llm_bridge.alignment_loss(
                projected_llm_emb[batch['anchor_ids']],
                temporal_entity_emb[batch['anchor_ids']],
            )

            # Total joint loss
            total_loss = (bpr_loss
                          + self.config.mu1 * ssl_loss
                          + self.config.mu2 * align_loss)
            return total_loss, bpr_loss, ssl_loss, align_loss

        elif mode == 'cold_start':
            # C4: Schema-conditioned MAML adaptation for new users
            adapted_user_emb = self.meta_bridge.adapt(
                support_set=batch['support_set'],
                kg_schema_stats=batch['kg_schema_stats'],
                base_user_emb=user_emb,
            )
            pos_score = self.ranking_layer(adapted_user_emb, pos_item_emb,
                                           pos_llm_emb, pos_temp_emb)
            return pos_score

        else:  # eval
            return pos_score

    def get_all_item_scores(self, user_id, kg_snapshots, llm_embeddings, item_ids):
        """Inference: score all candidate items for a single user."""
        temporal_entity_emb = self.temporal_builder(kg_snapshots)
        projected_llm_emb = self.llm_bridge(llm_embeddings)
        gnn_entity_emb, _ = self.graph_prop(
            x=temporal_entity_emb,
            edge_index=kg_snapshots[-1]['edge_index'],
            edge_attr=kg_snapshots[-1]['edge_attr'],
        )

        user_emb = self.user_embeddings(user_id).unsqueeze(0)       # [1, d]
        item_emb = gnn_entity_emb[item_ids]                          # [N, d]
        llm_emb = projected_llm_emb[item_ids]                        # [N, d]
        temp_emb = temporal_entity_emb[item_ids]                     # [N, d]

        scores = self.ranking_layer(user_emb.expand_as(item_emb),
                                    item_emb, llm_emb, temp_emb)     # [N]
        return scores
