#!/usr/bin/env python3
"""
Night Cycle Self-Play Daemon (Wormhole-Path 3)

Orchestrates parallel L0 sudo gitbash worktrees for autonomous evolution.
Uses Chaos Monkey to inject topological entropy (bugs) into code in worktrees.
Dispatches sheaf-guardian logic (via transducer pulse) to resolve.
On success (low energy, consistent), harvests Shape Pair to shape_pairs.jsonl.
On batch of 100, invokes qlora_distillation_stub with DeltaLambda1 weighting.
Integrates topological_kv_governor for context during resolutions.
Spawns worktrees with git (wrap with sudo git-bash in real env).
"""

import os
import sys
import subprocess
import threading
import time
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from rotary_condition_state import HybridConditionStateTransducer
from oracle import harvest_geometry_resolution
from qlora_distillation_stub import qlora_distillation_stub
from topological_kv_governor import TopologicalKVGovernor
from ast_sheaf_scanner import scan_ast_to_rips_complex

REPO_ROOT = Path(__file__).resolve().parents[1]
L0_ROOT = Path(os.environ.get("SHEAF_REPO_ROOT", str(REPO_ROOT))).resolve()
WORKTREES_DIR = L0_ROOT / "worktrees"
SHAPE_PAIRS = REPO_ROOT / "shape_pairs.jsonl"
BATCH_SIZE = 100
PARALLEL = 12


def chaos_monkey_intent_injector(worktree: Path) -> Dict[str, bytes]:
    """Record exact originals before adding a reversible simulation marker."""
    originals = {}
    for target in sorted(worktree.rglob("*.py"))[:3]:
        original = target.read_bytes()
        originals[str(target)] = original
        target.write_bytes(original + b"\n# SHEAF_SIMULATED_MUTATION\n")
    return originals


def dispatch_sheaf_guardian(worktree: Path, context: str = "night-cycle", originals=None,
                            transducer=None) -> Dict[str, Any]:
    """Evaluate and restore only known simulation mutations, byte for byte."""
    transducer = transducer or HybridConditionStateTransducer(dim=8)
    originals = originals or {}
    if not originals:
        return {"verdict": "NO_MUTATION", "energy": 0.0, "success": False}
    target = Path(next(iter(originals)))
    root = worktree.resolve()
    if not target.resolve().is_relative_to(root):
        raise ValueError("Mutation target is outside its worktree")
    artifact = target.read_text(encoding="utf-8")
    pre = transducer.pulse_mid_activity_evaluation(artifact, context)
    original = originals[str(target)]
    post = transducer.pulse_mid_activity_evaluation(original.decode("utf-8"), context + "-restored")
    for filename, contents in originals.items():
        path = Path(filename)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Mutation target is outside its worktree")
        path.write_bytes(contents)
    delta = pre["energy"] - post["energy"]
    return {"verdict": post["verdict"], "energy": post["energy"], "pre_energy": pre["energy"],
            "success": post["verdict"] == "CONSISTENT" and delta > 0,
            "target": target.relative_to(worktree).as_posix(), "delta_lambda": delta,
            "simulation": True, "note": "Known mutation restored; no autonomous semantic repair is implemented."}


def inject_llm_component_violations(wt: Path) -> List[str]:
    """Inject component sync violations for LLM COMPONENT ANALYSIS FOR INTEGRATION target.
    Violations represent hot linkages / mismatches in the geometric prime 0-dim map:
    - heuristic/embedding stalk assignment (violates strict AST syntactic point-cloud)
    - model architecture (Mamba/MoE/transformer) vs 0-dim stalk representation + copresheaf/ternary
    - 70B dense claim vs 6GB UMA + CTNN + sparse ternary + KV governor + RLM
    - cross-component (WORMHOLE closed-loop, pi-integration, multiple Forge versions) not yet wired to transducer/harvester/night
    These are the 'massive high-density target' files for the swarm. Resolution = make them coherent with the Epistemic Bounds pipeline (scanner first, prune before L0_F, Gini gate, harvester on +Δλ1, hook, etc.).
    """
    vdir = wt / "llm_component_violations"
    vdir.mkdir(parents=True, exist_ok=True)
    created = []

    # Violation 1: legacy stalk assignment using heuristic embeddings (vs real AST geometry from ast_sheaf_scanner)
    (vdir / "legacy_stalk_assignment.py").write_text('''#!/usr/bin/env python3
"""VIOLATION: heuristic embedding for 0-dim stalks of local LLMs on disk.
Should use scan_ast_to_rips_complex on model loader / architecture code + prune + transducer.
This is a flat degenerate mapping, not the geometric prime map.
"""
def assign_0dim_stalks_llm(gguf_paths):
    # Legacy: embeddings / size / quant as "stalks" (no restriction maps, no AST, no Morse)
    stalks = []
    for p in gguf_paths:
        stalks.append({"name": str(p), "embedding": [0.1, 0.2, 0.3]})  # toy, not syntactic
    return stalks
''', encoding='utf-8')
    created.append(str(vdir / "legacy_stalk_assignment.py"))

    # Violation 2: model-type stalk conflict (Mamba vs MoE vs transformer) not respecting copresheaf / ternary / 0-dim F(v)
    (vdir / "model_type_stalk_conflict.py").write_text('''#!/usr/bin/env python3
"""VIOLATION: Mamba / MoE / transformer architectures described as independent flat stalks.
Correct: each model's loading code / architecture is a syntactic complex; restriction maps F(v) -> F(e) via call/data deps;
pruned Morse boundary for UMA; only coherent H0 sections survive for the geometric map + CTNN copresheaf.
"""
def describe_mamba_as_stalk():
    return "Mamba SSM layers as 0-dim list (no copresheaf, no ternary routing, ignores 6GB UMA governor)"

def describe_moe_as_stalk():
    return "MoE experts as 0-dim list (dense claim, no L_F projection, no RLM for archive context)"

def describe_transformer_as_stalk():
    return "Transformer blocks as 0-dim (no integration with night/harvester for component analysis)"
''', encoding='utf-8')
    created.append(str(vdir / "model_type_stalk_conflict.py"))

    # Violation 3: 70B UMA / compression claim that violates the ZULU_YOKOHAMA payload (dense vs ternary sparse + CTNN + KV + RLM)
    (vdir / "uma_70b_dense_claim.py").write_text('''#!/usr/bin/env python3
"""VIOLATION: claims full 70B dense load possible in 6GB UMA without sparsification.
Correct per Epistemic Bounds / 70B spec: Asymmetric Mixed-Precision + SheafLaplacian + SVD for 18% H0 preservation at Q8_0/F16 + QPEFT/SRR;
Ternary Crystal {0 Void, 1 Identity, 3 Prime (MMA)}; copresheaf topological NN; topological_kv_governor L_F eviction + RLM for archive 6889-file context;
only sparse csr, prune before any L0_F, H0_Maintainers locked.
"""
def can_load_70b_dense_in_6gb():
    return "yes (dense, no governor, no ternary, no CTNN, no RLM chunking)"  # obstruction

def zulu_ternary_uma_correct():
    return "6GB UMA via sparse ternary + CTNN copresheaf + KV governor + RLM for massive LLM component archive"
''', encoding='utf-8')
    created.append(str(vdir / "uma_70b_dense_claim.py"))

    # Violation 4: cross-component hot linkage gap (WORMHOLE / pi / multiple Forge versions not wired to harvester / transducer / night for LLM integration)
    (vdir / "component_integration_gap.py").write_text('''#!/usr/bin/env python3
"""VIOLATION: powerful LLM components (Forge prompts/presets, WORMHOLE KIM/router/search/executor, pi-integration, multiple dashboard Forge/mcp/skill versions, atft engines)
exist but not yet coherently integrated via the night cycle targeting LLM components, the harvester for +Δλ1 Shape Pairs on the component manifold,
the transducer pulse as falsification for every proposed wiring, the mcp for assign_0dim_stalks + analyze_llm_components, the hook for zero-bypass edits.
This is exactly the hot linkage the swarm (with target=llm_component) is meant to resolve and distill into the local GGUF stalks.
"""
# Hot linkages (from the 0-dim structure map):
# - WORMHOLE closed-loop not yet calling harvest_geometry_resolution
# - pi-integration not wired to pre-tool-use hook or sheaf-condition-mcp analyze tools
# - multiple Forge versions (v1/v2/Zero/sheaf-stack) not unified under transducer pulse for coherence
# - night not yet defaulting to llm_component target against the archive + plugin LLM components
''', encoding='utf-8')
    created.append(str(vdir / "component_integration_gap.py"))

    return created


def create_worktree(repo_root: Path, worktree: Path):
    """Fail before any worker activity if git cannot create a real checkout."""
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                   cwd=str(repo_root), check=True, capture_output=True, text=True)


def run_night_cycle(cycles: int = 3, parallel: int = PARALLEL, batch: int = BATCH_SIZE, target: str = "llm_component"):
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import mkdtemp
    if min(cycles, parallel, batch) < 1:
        raise ValueError("cycles, parallel, batch must be positive")
    if target not in {"llm_component", "wellbore", "wellbore_example"}:
        raise ValueError("Unknown target")
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    harvested = 0

    def worker(worktree):
        transducer = HybridConditionStateTransducer(dim=8)
        governor = TopologicalKVGovernor(transducer=transducer)
        if target == "llm_component": inject_llm_component_violations(worktree)
        originals = chaos_monkey_intent_injector(worktree)
        result = dispatch_sheaf_guardian(worktree, originals=originals, transducer=transducer)
        result["governor"] = governor.govern(["simulation", "context"], current_vram_gb=5.7)
        return result

    for cycle in range(cycles):
        created = []
        try:
            for index in range(parallel):
                path = Path(mkdtemp(prefix=f"L0-night-{cycle}-{index}-", dir=WORKTREES_DIR))
                try:
                    create_worktree(L0_ROOT, path)
                except Exception:
                    path.rmdir()  # only this just-created, empty directory
                    raise
                created.append(path)
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                # Consume results on one thread; worker exceptions propagate.
                for result in pool.map(worker, created):
                    if result.get("success") and result["delta_lambda"] > 0:
                        receipt = harvest_geometry_resolution(
                            {"energy": result["pre_energy"]}, result,
                            result["delta_lambda"], [result["target"]])
                        if receipt.get("action") == "geometry_harvested": harvested += 1
            if harvested >= batch:
                # The distillation implementation is a stub, so preserve its
                # actual receipt instead of reporting that weights changed.
                receipt = qlora_distillation_stub(str(SHAPE_PAIRS), "local-llm.gguf")
                print({"distillation": receipt, "simulation": True})
                harvested = 0
        finally:
            for path in created:
                subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                               cwd=str(L0_ROOT), check=True, capture_output=True, text=True)
    return {"harvested": harvested, "simulation": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Isolated simulation of reversible mutations; no model training")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--target", default="llm_component", choices=["llm_component", "wellbore", "wellbore_example"])
    args = parser.parse_args()
    print(run_night_cycle(args.cycles, args.parallel, args.batch, args.target))
