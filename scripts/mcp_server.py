#!/usr/bin/env python3
"""Shared FastMCP and minimal stdio tool surface for the simulation harness."""
from __future__ import annotations
import contextlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))
from rotary_condition_state import pulse, read_state, TRANSDUCER
from oracle import handle_obstruction, ORACLE
from discrete_morse import prune_stalk_complex
from sheaf_svd_quantizer import sheaf_svd_quantize, load_synthetic_model
import hardware_piping as hw_piping


def rotate_condition(artifact: str, context: str = "agent-generation") -> Dict[str, Any]:
    result = pulse(artifact, context)
    decision = dict(result, pulse_summary=result, consistency_radius=TRANSDUCER.ComputeConsistencyRadius())
    if result["verdict"] != "CONSISTENT":
        decision["oracle_action"] = handle_obstruction(result["energy"], result.get("obstruction"), list(TRANSDUCER.state.stalks))
        decision["recommendation"] = "DO NOT EMIT until obstruction resolved."
    return decision


def read_condition_state() -> Dict[str, Any]:
    return dict(read_state(), oracle=ORACLE.get_stats(), transducer_class="HybridConditionStateTransducer")


def discrete_morse_prune(artifact: str = "", context: str = "prune-request", worktree: str = "") -> Dict[str, Any]:
    if artifact:
        return pulse(artifact, context, apply_discrete_morse=True)
    stalks, edges = TRANSDUCER.state.stalks, TRANSDUCER.state.edges
    reduced, reduced_edges, ratio = prune_stalk_complex(stalks, edges)
    return {"original_stalk_count": len(stalks), "original_edge_count": len(edges),
            "pruned_stalk_count": len(reduced), "pruned_edge_count": len(reduced_edges),
            "sparsification_ratio": ratio, "homology_preservation": "not_checked",
            "note": "Preview only. Original state is retained for the energy gate."}


def topological_kv_govern(tokens: list[str] | None = None, current_vram: float = 5.9) -> dict:
    from topological_kv_governor import TopologicalKVGovernor
    return TopologicalKVGovernor().govern(tokens or [], current_vram_gb=current_vram)


def _run_svd_quantize(model: str = "synthetic", hidden: int = 32, layers: int = 1, preserve_ratio: float = 0.18) -> Dict[str, Any]:
    if model != "synthetic":
        raise ValueError("MCP quantization supports only model='synthetic'; use the quantizer CLI for checkpoints")
    if not 1 <= hidden <= 64 or not 1 <= layers <= 4:
        raise ValueError("Synthetic MCP demo requires hidden in 1..64 and layers in 1..4")
    with contextlib.redirect_stdout(sys.stderr):
        qmap = sheaf_svd_quantize(load_synthetic_model(hidden, layers), preserve_ratio=preserve_ratio)
    out = Path(__file__).resolve().parents[1] / "config/ternary_mcp_ggufmap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qmap, indent=2), encoding="utf-8")
    return {"status": "synthetic_quantized", "model": model, "layers": len(qmap["layers"]),
            "map_path": str(out), "h0_maintainer_energy": qmap["h0_maintainer_energy"],
            "note": "Synthetic proxy map; custom llama.cpp integration is not implemented."}


run_sheaf_svd_quantize = _run_svd_quantize


def _apply_ternary_map(model_path: str = "", gguf_map_path: str = "config/ternary_mcp_ggufmap.json") -> Dict[str, Any]:
    qmap = json.loads(Path(gguf_map_path).read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(sys.stderr):
        bridge = hw_piping.LlamaCppTopologicalBridge(hw_piping.LlamaCppBridgeConfig(model_path=model_path))
        result = bridge.apply_ternary_gguf_map(qmap)
    return {"status": result["status"], "applied": result, "backend": bridge._backend_type}


apply_ternary_map_to_llama = _apply_ternary_map


def start_night_cycle(cycles: int = 2, parallel: int = 4, batch: int = 5) -> dict:
    from night_cycle_daemon import run_night_cycle
    if cycles < 1 or parallel < 1 or batch < 1:
        raise ValueError("cycles, parallel, and batch must be positive")
    threading.Thread(target=run_night_cycle, kwargs=dict(cycles=cycles, parallel=parallel, batch=batch), daemon=True).start()
    return {"status": "started", "simulation": True, "cycles": cycles, "parallel": parallel, "batch": batch}


TOOLS = {name: globals()[name] for name in (
    "rotate_condition", "read_condition_state", "discrete_morse_prune", "topological_kv_govern",
    "run_sheaf_svd_quantize", "apply_ternary_map_to_llama", "start_night_cycle")}


def _make_tool_response(result):
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


def _schema(function):
    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        default = parameter.default
        kind = "string"
        if name in {"hidden", "layers", "cycles", "parallel", "batch"}: kind = "integer"
        elif name in {"preserve_ratio", "current_vram"}: kind = "number"
        elif name == "tokens": kind = "array"
        properties[name] = {"type": kind}
        if kind == "array": properties[name]["items"] = {"type": "string"}
        if default is inspect.Parameter.empty: required.append(name)
        elif default is not None: properties[name]["default"] = default
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


class MinimalMCP:
    def __init__(self):
        self.tools = {name: {"description": (fn.__doc__ or name), "inputSchema": _schema(fn)} for name, fn in TOOLS.items()}

    def handle_request(self, req):
        rid, method = req.get("id"), req.get("method")
        if "id" not in req: return None
        result = None
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "sheaf-condition-mcp", "version": "0.1.0"}}
        elif method == "tools/list":
            result = {"tools": [dict(name=name, **meta) for name, meta in self.tools.items()]}
        elif method == "tools/call":
            params = req.get("params") or {}
            name, arguments = params.get("name"), params.get("arguments") or {}
            if name not in TOOLS:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"Unknown tool {name}"}}
            try:
                inspect.signature(TOOLS[name]).bind(**arguments)
                with contextlib.redirect_stdout(sys.stderr):
                    result = _make_tool_response(TOOLS[name](**arguments))
            except Exception as error:
                result = {"isError": True, "content": [{"type": "text", "text": str(error)}]}
        elif method == "ping": result = {}
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown method {method}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}


try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None


def main():
    if FastMCP is not None and os.environ.get("SHEAF_MINIMAL_MCP") != "1":
        mcp = FastMCP("sheaf-condition-mcp")
        for name, function in TOOLS.items(): mcp.tool(name=name)(function)
        mcp.run()
        return
    server = MinimalMCP()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict): raise ValueError("Request must be an object")
            response = server.handle_request(request)
        except (ValueError, TypeError) as error:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}}
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
