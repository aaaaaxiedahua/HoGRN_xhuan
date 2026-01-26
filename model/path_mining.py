"""
Path Mining Module for RPG-HoGRN

This module discovers frequent relation paths from the knowledge graph.
Paths are mined during preprocessing (no gradients needed).
"""

from collections import defaultdict
import pickle
import os


class PathMiner:
    """
    Relation Path Miner
    Scans training triples to find frequent 2-3 hop relation paths.
    """

    def __init__(self, triples, num_relations):
        """
        Args:
            triples: List of (head, relation, tail)
            num_relations: Number of relation types (without inverse)
        """
        self.triples = triples
        self.num_relations = num_relations

        print(f"[PathMiner] Initializing with {len(triples)} triples, {num_relations} relations")

        # Build adjacency list: entity -> [(relation, neighbor), ...]
        self.adj_out = defaultdict(list)  # outgoing edges
        self.adj_in = defaultdict(list)   # incoming edges (inverse)

        for h, r, t in triples:
            self.adj_out[h].append((r, t))
            self.adj_in[t].append((r, h))

        print(f"[PathMiner] Built adjacency: {len(self.adj_out)} entities with outgoing edges")

    def mine_paths(self, max_length=3, min_count=50):
        """
        Mine frequent relation paths.

        Args:
            max_length: Maximum path length (2 or 3)
            min_count: Minimum occurrence threshold

        Returns:
            path_count: Dict[(r1, r2, ...), count]
        """
        path_count = defaultdict(int)
        path_2_count = 0
        path_3_count = 0

        # Iterate all triples as starting point
        for h, r1, t1 in self.triples:
            # Length-2 paths: h -r1-> t1 -r2-> t2
            for r2, t2 in self.adj_out[t1]:
                path_count[(r1, r2)] += 1
                path_2_count += 1

                # Length-3 paths: h -r1-> t1 -r2-> t2 -r3-> t3
                if max_length >= 3:
                    for r3, t3 in self.adj_out[t2]:
                        path_count[(r1, r2, r3)] += 1
                        path_3_count += 1

        print(f"[PathMiner] Raw path instances: 2-hop={path_2_count}, 3-hop={path_3_count}")
        print(f"[PathMiner] Unique path patterns before filtering: {len(path_count)}")

        # Filter by frequency
        frequent_paths = {
            path: count
            for path, count in path_count.items()
            if count >= min_count
        }

        # Statistics
        path_2_filtered = sum(1 for p in frequent_paths if len(p) == 2)
        path_3_filtered = sum(1 for p in frequent_paths if len(p) == 3)
        print(f"[PathMiner] After filtering (min_count={min_count}): 2-hop={path_2_filtered}, 3-hop={path_3_filtered}")

        # Show top-5 frequent paths
        sorted_paths = sorted(frequent_paths.items(), key=lambda x: -x[1])[:5]
        print(f"[PathMiner] Top-5 frequent paths:")
        for path, count in sorted_paths:
            print(f"  {path} -> {count} times")

        return frequent_paths

    def build_path_index(self, frequent_paths):
        """
        Build index: first_relation -> list of paths starting with it.

        Args:
            frequent_paths: Dict of frequent paths

        Returns:
            rel_to_paths: Dict[rel] -> [(path, count), ...]
        """
        rel_to_paths = defaultdict(list)

        for path, count in frequent_paths.items():
            first_rel = path[0]
            rel_to_paths[first_rel].append((path, count))

        # Sort by frequency (descending)
        for rel in rel_to_paths:
            rel_to_paths[rel].sort(key=lambda x: -x[1])

        return dict(rel_to_paths)


def mine_and_save_paths(triples, num_relations, save_dir,
                        max_length=3, min_count=50):
    """
    Mine paths and save to disk.

    Args:
        triples: List of (h, r, t)
        num_relations: Number of relations
        save_dir: Directory to save results
        max_length: Max path length
        min_count: Min frequency threshold

    Returns:
        frequent_paths, rel_to_paths
    """
    miner = PathMiner(triples, num_relations)

    print(f"[PathMiner] Mining paths (max_len={max_length}, min_count={min_count})...")
    frequent_paths = miner.mine_paths(max_length, min_count)
    print(f"[PathMiner] Found {len(frequent_paths)} frequent paths")

    rel_to_paths = miner.build_path_index(frequent_paths)
    print(f"[PathMiner] Built index for {len(rel_to_paths)} relations")

    # Save to disk
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'frequent_paths.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump({
            'frequent_paths': frequent_paths,
            'rel_to_paths': rel_to_paths,
            'max_length': max_length,
            'min_count': min_count
        }, f)

    print(f"[PathMiner] Saved to {save_path}")

    return frequent_paths, rel_to_paths


def load_paths(save_dir):
    """Load pre-mined paths from disk."""
    save_path = os.path.join(save_dir, 'frequent_paths.pkl')

    if not os.path.exists(save_path):
        return None, None

    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    print(f"[PathMiner] Loaded {len(data['frequent_paths'])} paths from {save_path}")
    return data['frequent_paths'], data['rel_to_paths']
