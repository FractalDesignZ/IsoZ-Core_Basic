"""
Discrete Morse Theory for Exact Sparsification.

Implements conservative elementary collapses for one-dimensional graphs.

Used to prune Vietoris-Rips / AST-derived simplicial complexes before Sheaf Laplacian computation,
Graph homology is preserved by free-face collapses. Sheaf energy preservation is not established.

This fulfills Node α/β of the Epistemic Bounds for Oracle-speed on pruned critical cells.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy import sparse

@dataclass
class SimplicialComplex:
    """Lightweight simplicial complex representation."""
    cells: Dict[int, List[int]] = field(default_factory=dict)  # dim -> list of cell ids
    faces: Dict[int, List[int]] = field(default_factory=dict)  # cell_id -> face ids
    vertices: List[int] = field(default_factory=list)
    dim: int = 0

@dataclass
class DiscreteVectorField:
    """Discrete vector field: pairs lower -> upper cell."""
    matching: Dict[int, int] = field(default_factory=dict)
    reverse: Dict[int, int] = field(default_factory=dict)

def build_from_stalks_and_edges(stalks: Dict[str, dict], edges: List[Tuple[str, str]]) -> Tuple[SimplicialComplex, Dict[str, int]]:
    """Build a 1-complex from stalks (0-cells) and edges (1-cells)."""
    complex = SimplicialComplex()
    id_map = {sid: i for i, sid in enumerate(stalks.keys())}
    complex.vertices = list(range(len(stalks)))
    complex.cells[0] = list(range(len(stalks)))
    for u, v in edges:
        if u in id_map and v in id_map:
            eid = len(stalks) + len(complex.faces)
            complex.cells.setdefault(1, []).append(eid)
            complex.faces[eid] = [id_map[u], id_map[v]]
    complex.dim = 1 if complex.cells.get(1) else 0
    return complex, id_map

def compute_discrete_gradient(complex: SimplicialComplex, filtration: Optional[np.ndarray] = None) -> DiscreteVectorField:
    """Match only free vertex/edge pairs in a graph.

    Removing a degree-one vertex and its sole edge is an elementary collapse.
    Recompute degrees after every collapse; cyclic cores are left intact.
    Higher dimensional inputs are left unchanged until signed V-path transport
    is implemented.
    """
    vf = DiscreteVectorField()
    if complex.dim > 1:
        return vf
    remaining = {eid: list(complex.faces[eid]) for eid in complex.cells.get(1, [])}
    while True:
        cofaces = {}
        for eid, vertices in remaining.items():
            if len(vertices) == 2:
                for vertex in vertices:
                    cofaces.setdefault(vertex, []).append(eid)
        free = next(((v, es[0]) for v, es in sorted(cofaces.items()) if len(es) == 1), None)
        if free is None:
            break
        vertex, edge = free
        vf.matching[vertex] = edge
        vf.reverse[edge] = vertex
        del remaining[edge]
    return vf


def extract_critical_cells(complex: SimplicialComplex, vf: DiscreteVectorField) -> Dict[int, List[int]]:
    critical: Dict[int, List[int]] = {}
    for dim, cells in complex.cells.items():
        crit = [c for c in cells if c not in vf.matching and c not in vf.reverse]
        if crit:
            critical[dim] = crit
    return critical

def build_morse_complex(complex: SimplicialComplex, vf: DiscreteVectorField, critical: Dict[int, List[int]]) -> Tuple[SimplicialComplex, sparse.csr_matrix]:
    """Build the residual graph and its actual oriented incidence boundary."""
    morse = SimplicialComplex()
    morse.vertices = list(critical.get(0, []))
    morse.cells = {d: list(cs) for d, cs in critical.items()}
    morse.dim = max(morse.cells, default=0)
    morse.faces = {cell: list(complex.faces[cell]) for cells in morse.cells.values()
                   for cell in cells if cell in complex.faces}
    all_cells = [cell for dim in sorted(morse.cells) for cell in morse.cells[dim]]
    indices = {cell: i for i, cell in enumerate(all_cells)}
    rows, cols, data = [], [], []
    for edge in morse.cells.get(1, []):
        for sign, vertex in zip((-1.0, 1.0), morse.faces[edge]):
            rows.append(indices[vertex]); cols.append(indices[edge]); data.append(sign)
    boundary = sparse.csr_matrix((data, (rows, cols)), shape=(len(all_cells), len(all_cells)))
    return morse, boundary


def prune_complex_to_critical(complex: SimplicialComplex, filtration: Optional[np.ndarray] = None) -> Tuple[SimplicialComplex, sparse.csr_matrix, Dict[int, List[int]]]:
    """Full pruning pipeline: gradient -> matching -> critical cells -> Morse complex + boundary."""
    if complex.dim > 1:
        raise NotImplementedError("Only graph complexes are supported")
    vf = compute_discrete_gradient(complex, filtration)
    critical = extract_critical_cells(complex, vf)
    morse, boundary = build_morse_complex(complex, vf, critical)
    return morse, boundary, critical

def prune_stalk_complex(stalks: Dict[str, dict], edges: List[Tuple[str, str]]) -> Tuple[Dict[str, dict], List[Tuple[str, str]], float]:
    """Convenience: prune the current transducer's stalks/edges using Discrete Morse.
    Returns pruned_stalks, pruned_edges, reduction_ratio.
    """
    if len(stalks) < 3:
        return stalks, edges, 1.0
    complex, id_map = build_from_stalks_and_edges(stalks, edges)
    morse, boundary, critical = prune_complex_to_critical(complex)
    # Map back
    rev_map = {v: k for k, v in id_map.items()}
    pruned_stalk_ids = [rev_map[i] for i in critical.get(0, [])]
    pruned_stalks = {sid: stalks[sid] for sid in pruned_stalk_ids if sid in stalks}
    # Rebuild edges among pruned (simplified: keep original if both ends pruned)
    pruned_edges = [e for e in edges if e[0] in pruned_stalks and e[1] in pruned_stalks]
    reduction = len(pruned_stalks) / len(stalks) if stalks else 1.0
    return pruned_stalks, pruned_edges, reduction

if __name__ == "__main__":
    # Demo
    stalks = {f"s{i}": {"vec": [i]} for i in range(20)}
    edges = [(f"s{i}", f"s{i+1}") for i in range(19)]
    p_stalks, p_edges, ratio = prune_stalk_complex(stalks, edges)
    print(f"Pruned from {len(stalks)} to {len(p_stalks)} stalks, ratio {ratio:.2f}")