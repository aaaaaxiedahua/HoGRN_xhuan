"""
Prototype Enhancement Module

Core idea: for each relation, compute an "answer prototype" (mean embedding of all
answer entities), then score candidates by similarity to the prototype.
- 100% coverage: every entity gets a continuous similarity score
- Evolves with training: prototypes update as GCN embeddings improve
- Minimal overhead: one matrix-vector multiply per relation per epoch
"""

import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PrototypeEnhancer(nn.Module):

    def __init__(self, num_relations, num_ent, embed_dim):
        super().__init__()
        self.num_relations = num_relations
        self.num_ent = num_ent
        self.embed_dim = embed_dim
        self.num_rel_total = num_relations * 2  # including inverse

        # Learnable fusion coefficient (sigmoid(-3) ~ 0.05, starts small)
        self.gamma = nn.Parameter(torch.tensor(-3.0))

        # Answer entity IDs for each relation: list of LongTensor
        self.answer_ids = [None] * self.num_rel_total

        # Cached prototype scores: [num_rel_total, N]
        self.register_buffer('proto_scores', torch.zeros(self.num_rel_total, num_ent))

        self.built = False
        self.current_epoch = -1
        self.forward_count = 0
        self.log_interval = 100

        logger.info(f"[PrototypeEnhancer] num_rel={num_relations}, "
                    f"num_ent={num_ent}, embed_dim={embed_dim}")

    def build_answer_sets(self, edge_index, edge_type):
        """Collect answer entity IDs for each relation (one-time)."""
        logger.info("[PrototypeEnhancer] Building answer sets...")

        answer_sets = defaultdict(set)
        edge_dst = edge_index[1].cpu().numpy()
        edge_tp = edge_type.cpu().numpy()

        for i in range(len(edge_tp)):
            answer_sets[int(edge_tp[i])].add(int(edge_dst[i]))

        for r in range(self.num_rel_total):
            if r in answer_sets and len(answer_sets[r]) > 0:
                self.answer_ids[r] = torch.LongTensor(list(answer_sets[r]))
            else:
                self.answer_ids[r] = None

        covered = sum(1 for ids in self.answer_ids if ids is not None)
        total_answers = sum(len(ids) for ids in self.answer_ids if ids is not None)
        logger.info(f"[PrototypeEnhancer] {covered}/{self.num_rel_total} relations have answers, "
                    f"total answer entities: {total_answers}")

        # Log answer set sizes for first 5 relations
        for r in range(min(5, self.num_rel_total)):
            sz = len(self.answer_ids[r]) if self.answer_ids[r] is not None else 0
            logger.info(f"  Rel {r}: {sz} answer entities")

        self.built = True

    def update_prototypes(self, all_ent, epoch):
        """
        Compute prototypes and similarity scores using current embeddings.
        Called once per epoch.

        Args:
            all_ent: [N, dim] entity embeddings from GCN (detached)
            epoch: current epoch number
        """
        if epoch == self.current_epoch:
            return  # already updated this epoch

        self.current_epoch = epoch
        device = all_ent.device

        # Normalize entity embeddings for cosine similarity
        all_ent_norm = F.normalize(all_ent, p=2, dim=1)  # [N, dim]

        update_count = 0
        for r in range(self.num_rel_total):
            if self.answer_ids[r] is None:
                self.proto_scores[r].zero_()
                continue

            ids = self.answer_ids[r].to(device)
            # Prototype = mean of answer entity embeddings (normalized)
            proto = all_ent_norm[ids].mean(dim=0)  # [dim]
            proto = F.normalize(proto, p=2, dim=0)  # normalize prototype

            # Cosine similarity with all entities
            sim = torch.mv(all_ent_norm, proto)  # [N]
            self.proto_scores[r] = sim
            update_count += 1

        if epoch % 10 == 0 or epoch == 0:
            avg_score = self.proto_scores.mean().item()
            std_score = self.proto_scores.std().item()
            logger.info(f"[PrototypeEnhancer] Epoch {epoch}: updated {update_count} prototypes, "
                        f"avg_sim={avg_score:.4f}, std_sim={std_score:.4f}")

    def forward(self, original_score, query_rel, all_ent, epoch):
        """
        Enhance original scores with prototype similarity.

        Args:
            original_score: [batch, N] raw scores before sigmoid
            query_rel: [batch] relation indices
            all_ent: [N, dim] current entity embeddings
            epoch: current epoch number

        Returns:
            final_score: [batch, N]
        """
        # Build answer sets on first call
        if not self.built:
            self.build_answer_sets(
                self._edge_index, self._edge_type
            )

        # Update prototypes once per epoch (use detached embeddings)
        self.update_prototypes(all_ent.detach(), epoch)

        # Lookup pre-computed scores (no gradient)
        pattern_score = self.proto_scores[query_rel]  # [batch, N]

        # Fusion
        gamma = torch.sigmoid(self.gamma)
        final_score = original_score + gamma * pattern_score

        # Log
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            gamma_val = gamma.item()
            avg_ps = pattern_score.mean().item()
            boost = (gamma * pattern_score).mean().item()
            orig_mean = original_score.mean().item()
            logger.info(f"[PrototypeEnhancer] Step {self.forward_count}: "
                        f"gamma={gamma_val:.4f}, "
                        f"avg_sim={avg_ps:.4f}, "
                        f"avg_boost={boost:.4f}, "
                        f"orig_mean={orig_mean:.4f}")

        return final_score
