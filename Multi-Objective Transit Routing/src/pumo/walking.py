"""Bounded Pareto walking searches over the prepared directed pedestrian graph."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WalkOption:
    target: int
    walk_time_s: int
    distance_m: float
    risk_exposure: float
    edge_ids: tuple[int, ...]


class WalkingGraph:
    """Compact CSR-like arrays; IDs are dense prepared-network node indices."""
    def __init__(self, node_count: int, from_node, to_node, edge_id, time_s, distance_m, risk):
        from_node = np.asarray(from_node, dtype=np.int64)
        to_node = np.asarray(to_node, dtype=np.int64)
        order = np.argsort(from_node, kind="stable")
        self.node_count = int(node_count)
        counts = np.bincount(from_node, minlength=self.node_count)
        self.indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.to = to_node[order]
        self.edge = np.asarray(edge_id, dtype=np.int64)[order]
        self.time = np.asarray(time_s, dtype=np.int32)[order]
        self.distance = np.asarray(distance_m, dtype=np.float32)[order]
        self.risk = np.asarray(risk, dtype=np.float32)[order]

    def bounded_options(self, source: int, targets: Iterable[int], *, max_distance_m: float = 400,
                        max_time_s: int = 420, limit: int = 3,
                        labels_per_node: int = 12) -> list[WalkOption]:
        """Return up to ``limit`` non-dominated walks to each requested node.

        The bounded per-node label set limits work on real city graphs. If it is
        reached, candidates are retained by a deterministic normalized score;
        callers should record this configured approximation cap.
        """
        targets = set(int(x) for x in targets)
        targets.discard(int(source))
        if not targets:
            return []
        # Labels: (node, time, distance, risk, parent label id, incoming edge id).
        labels: list[list] = [[int(source), 0, 0.0, 0.0, -1, -1]]
        active: dict[int, list[int]] = {int(source): [0]}
        queue = [(0, 0)]
        while queue:
            _, label_id = heapq.heappop(queue)
            node, t0, d0, r0, _, _ = labels[label_id]
            if label_id not in active.get(node, ()):
                continue
            for pos in range(int(self.indptr[node]), int(self.indptr[node + 1])):
                nxt = int(self.to[pos])
                t = t0 + int(self.time[pos]); d = d0 + float(self.distance[pos]); r = r0 + float(self.risk[pos])
                if t > max_time_s or d > max_distance_m:
                    continue
                candidates = active.setdefault(nxt, [])
                if any(labels[i][1] <= t and labels[i][2] <= d + 1e-6 and labels[i][3] <= r + 1e-6
                       for i in candidates):
                    continue
                survivors = [i for i in candidates if not (t <= labels[i][1] and d <= labels[i][2] + 1e-6
                            and r <= labels[i][3] + 1e-6)]
                new_id = len(labels)
                labels.append([nxt, t, d, r, label_id, int(self.edge[pos])])
                survivors.append(new_id)
                if len(survivors) > labels_per_node:
                    survivors.sort(key=lambda i: (labels[i][1] / max_time_s + labels[i][2] / max_distance_m +
                                                   labels[i][3] / max(1.0, r + 1.0), labels[i][1], labels[i][2], labels[i][3]))
                    survivors = survivors[:labels_per_node]
                active[nxt] = survivors
                if new_id in survivors:
                    heapq.heappush(queue, (t, new_id))
        result = []
        for target in sorted(targets):
            ids = active.get(target, [])
            ids = sorted(ids, key=lambda i: (labels[i][1], labels[i][2], labels[i][3]))[:limit]
            for i in ids:
                path = []
                cursor = i
                while labels[cursor][4] >= 0:
                    path.append(labels[cursor][5])
                    cursor = labels[cursor][4]
                path.reverse()
                result.append(WalkOption(target, int(labels[i][1]), float(labels[i][2]),
                                         float(labels[i][3]), tuple(path)))
        return result
