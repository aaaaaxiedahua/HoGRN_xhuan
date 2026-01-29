"""
RPG-HoGRN: Reverse Path Reasoning Module

Core idea: learn "answer patterns" for each relation, check if candidate entities match
- Offline: for each relation r, count in/out edge pattern frequency of its answer entities
- Online: compute pattern match score for all candidates, add to original score
- Does not depend on head->tail path connectivity, only looks at tail's local structure
"""

import logging
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ReversePathReasoner(nn.Module):

    def __init__(self, num_relations, num_ent):
        super().__init__()
        self.num_relations = num_relations
        self.num_ent = num_ent
        self.num_patterns = 2 * num_relations * 2  # in/out x all relations (incl. inverse)

        # Fixed relation-pattern weights (buffer, no gradient)
        self.register_buffer('pattern_weight', torch.zeros(num_relations * 2, self.num_patterns))

        # Fixed entity-pattern matrix (buffer, no gradient)
        self.register_buffer('entity_pattern_matrix', torch.zeros(num_ent, self.num_patterns))

        # Pre-computed pattern scores per relation: [num_rel*2, N]
        self.register_buffer('rel_pattern_scores', None)

        # Only gamma is learnable (small init so module starts with minimal influence)
        self.gamma = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ~ 0.05

        self.built = False
        self.forward_count = 0
        self.log_interval = 100

        logger.info(f"[ReversePathReasoner] num_relations={num_relations}, "
                    f"num_ent={num_ent}, num_patterns={self.num_patterns}")

    def _pattern_index(self, direction, rel_id):
        return direction * (self.num_relations * 2) + rel_id

    def build(self, edge_index, edge_type, device):
        logger.info("[ReversePathReasoner] Building...")

        num_rel_total = self.num_relations * 2

        edge_src = edge_index[0].cpu().numpy()
        edge_dst = edge_index[1].cpu().numpy()
        edge_tp = edge_type.cpu().numpy()
        num_edges = len(edge_tp)

        # Step 1: Build adjacency
        in_edges = defaultdict(list)
        out_edges = defaultdict(list)

        for i in range(num_edges):
            src, dst, r = int(edge_src[i]), int(edge_dst[i]), int(edge_tp[i])
            out_edges[src].append(r)
            in_edges[dst].append(r)

        logger.info(f"[ReversePathReasoner] {len(out_edges)} entities with out-edges, "
                    f"{len(in_edges)} entities with in-edges")

        # Step 2: Build entity-pattern matrix M[N, P] = log(1 + count)
        entity_pattern = torch.zeros(self.num_ent, self.num_patterns)

        for entity, rels in in_edges.items():
            rel_count = defaultdict(int)
            for r in rels:
                rel_count[r] += 1
            for r, count in rel_count.items():
                entity_pattern[entity, self._pattern_index(0, r)] = torch.log1p(torch.tensor(float(count)))

        for entity, rels in out_edges.items():
            rel_count = defaultdict(int)
            for r in rels:
                rel_count[r] += 1
            for r, count in rel_count.items():
                entity_pattern[entity, self._pattern_index(1, r)] = torch.log1p(torch.tensor(float(count)))

        self.entity_pattern_matrix.copy_(entity_pattern.to(device))

        nonzero_count = (entity_pattern > 0).sum().item()
        total_count = entity_pattern.numel()
        logger.info(f"[ReversePathReasoner] Entity-pattern matrix: "
                    f"nonzero={nonzero_count}/{total_count} ({100*nonzero_count/total_count:.2f}%)")

        # Step 3: Collect answer sets and compute frequency
        answer_sets = defaultdict(set)
        for i in range(num_edges):
            answer_sets[int(edge_tp[i])].add(int(edge_dst[i]))

        logger.info(f"[ReversePathReasoner] Answer sets for {len(answer_sets)} relations")

        freq_matrix = torch.zeros(num_rel_total, self.num_patterns)
        for r, answers in answer_sets.items():
            if len(answers) == 0:
                continue
            n_answers = len(answers)
            for p_idx in range(self.num_patterns):
                count = sum(1 for ans in answers if entity_pattern[ans, p_idx] > 0)
                freq_matrix[r, p_idx] = count / n_answers

        self.pattern_weight.copy_(freq_matrix.to(device))

        # Step 4: Pre-compute pattern scores for each relation
        # rel_scores[r] = freq_matrix[r] @ entity_pattern_matrix.T -> [N]
        # Then normalize per relation to [0, 1]
        raw_scores = torch.mm(freq_matrix.to(device), self.entity_pattern_matrix.t())  # [R, N]

        # Per-relation min-max normalization
        s_min = raw_scores.min(dim=1, keepdim=True)[0]
        s_max = raw_scores.max(dim=1, keepdim=True)[0]
        self.rel_pattern_scores = (raw_scores - s_min) / (s_max - s_min + 1e-8)  # [R, N]

        # Stats
        avg_patterns = (freq_matrix > 0).sum(dim=1).float().mean().item()
        avg_score = self.rel_pattern_scores.mean().item()
        max_score = self.rel_pattern_scores.max().item()
        logger.info(f"[ReversePathReasoner] Avg patterns/rel: {avg_patterns:.1f}, "
                    f"avg_precomputed_score: {avg_score:.4f}, max: {max_score:.4f}")

        # Top patterns for first 5 relations
        for r in range(min(5, num_rel_total)):
            weights = freq_matrix[r]
            nonzero_num = (weights > 0).sum().item()
            if nonzero_num > 0:
                top_vals, top_idxs = torch.topk(weights, min(3, nonzero_num))
                patterns_str = []
                for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
                    direction = "in" if idx < num_rel_total else "out"
                    rel_id = idx % num_rel_total
                    patterns_str.append(f"({direction},r{rel_id}):{val:.2f}")
                logger.info(f"  Rel {r}: {', '.join(patterns_str)}")

        self.built = True
        logger.info("[ReversePathReasoner] Build complete!")

    def forward(self, original_score, query_rel, edge_index, edge_type):
        device = original_score.device

        if not self.built:
            self.build(edge_index=edge_index, edge_type=edge_type, device=device)

        # Lookup pre-computed scores (no gradient through pattern computation)
        # query_rel: [batch]
        # rel_pattern_scores: [R, N]
        pattern_score = self.rel_pattern_scores[query_rel]  # [batch, N], no grad

        # Only gamma is learnable
        gamma = torch.sigmoid(self.gamma)  # starts at ~0.05
        final_score = original_score + gamma * pattern_score

        # Log
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            gamma_val = gamma.item()
            avg_ps = pattern_score.mean().item()
            boost = (gamma * pattern_score).mean().item()
            orig_mean = original_score.mean().item()
            logger.info(f"[ReversePathReasoner] Step {self.forward_count}: "
                        f"gamma={gamma_val:.4f}, "
                        f"avg_pattern={avg_ps:.4f}, "
                        f"avg_boost={boost:.4f}, "
                        f"orig_mean={orig_mean:.4f}")

        return final_score
