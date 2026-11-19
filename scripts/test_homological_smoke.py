#!/usr/bin/env python3
"""Fail on graph homology or original-graph obstruction regressions.

This checks small synthetic graphs, not a general sheaf preservation theorem.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from discrete_morse import build_from_stalks_and_edges, prune_complex_to_critical
from rotary_condition_state import HybridConditionStateTransducer


def graph_betti(complex):
    parent = {vertex: vertex for vertex in complex.vertices}
    def find(vertex):
        while parent[vertex] != vertex:
            vertex = parent[vertex]
        return vertex
    for edge in complex.cells.get(1, []):
        u, v = complex.faces[edge]
        parent[find(u)] = find(v)
    components = len({find(vertex) for vertex in parent})
    return components, len(complex.cells.get(1, [])) - len(parent) + components


def main():
    for edges in [[("a", "b"), ("b", "c")], [("a", "b"), ("b", "c"), ("c", "a")], [("a", "a"), ("a", "b")]]:
        complex, _ = build_from_stalks_and_edges({key: {} for key in "abcd"}, edges)
        reduced, boundary, _ = prune_complex_to_critical(complex)
        assert graph_betti(complex) == graph_betti(reduced), "Graph Betti numbers changed"
        assert np.allclose((boundary @ boundary).toarray(), 0), "Boundary does not square to zero"
    source = "def outer():\n    inner()\n    return other()\n"
    baseline = HybridConditionStateTransducer().pulse_mid_activity_evaluation(source, apply_discrete_morse=False)
    pruned = HybridConditionStateTransducer().pulse_mid_activity_evaluation(source, apply_discrete_morse=True)
    assert baseline["verdict"] == pruned["verdict"] == "OBSTRUCTED"
    assert baseline["energy"] == pruned["energy"] > 0
    assert baseline["hot_linkages"] == pruned["hot_linkages"]
    assert not baseline["sparsification_applied"] and pruned["sparsification_applied"]
    print("PASS: synthetic graph homology and original-graph obstruction checks")


if __name__ == "__main__":
    main()
