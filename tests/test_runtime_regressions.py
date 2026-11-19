import ast
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ast_sheaf_scanner as scanner
import discrete_morse as morse
import hardware_piping as hardware
import mcp_server as mcp
import night_cycle_daemon as night
import oracle
import rotary_condition_state as rotary
import sheaf_coderabbit as reviewer
import sheaf_svd_quantizer as quantizer
import topological_kv_governor as kv
from test_homological_smoke import graph_betti
from tool_use_hook import proposed_artifact


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rotary, "harvest_geometry_resolution", Mock(return_value={"action": "test_only"}))
    rotary.TRANSDUCER.reset()


@pytest.mark.parametrize("edges", [[], [("a","b"),("b","c")], [("a","b"),("b","c"),("c","a")], [("a","a"),("a","b")], [("a","b"),("a","b")]])
def test_graph_collapses_preserve_betti(edges):
    complex, mapping = morse.build_from_stalks_and_edges({x:{} for x in "abcd"}, edges)
    reduced, boundary, _ = morse.prune_complex_to_critical(complex)
    assert graph_betti(complex) == graph_betti(reduced)
    assert (boundary @ boundary).nnz == 0
    assert reduced.vertices


def test_large_vertex_catalog_uses_disjoint_cell_ids():
    complex, _ = morse.build_from_stalks_and_edges({str(i):{} for i in range(1002)}, [("0","1001")])
    assert not set(complex.cells[0]) & set(complex.cells[1])


def test_nested_function_restores_caller():
    complex, _, names = scanner.extract_topological_features(ast.parse("def outer():\n def inner(): child()\n after()\n"))
    edges = {(names[u], names[v]) for u,v in complex.faces.values()}
    assert ("outer","after") in edges and ("inner","child") in edges


def test_ast_energy_and_morse_flag_and_json():
    source = "def outer():\n first()\n return second()\n"
    baseline = rotary.HybridConditionStateTransducer().pulse_mid_activity_evaluation(source, apply_discrete_morse=False)
    t = rotary.HybridConditionStateTransducer()
    reduced = t.pulse_mid_activity_evaluation(source, apply_discrete_morse=True)
    assert reduced["energy"] == baseline["energy"] > 0
    assert reduced["verdict"] == baseline["verdict"] == "OBSTRUCTED"
    assert not baseline["sparsification_applied"]
    assert reduced["original_stalk_count"] > reduced["pruned_stalk_count"]
    assert all(u in t.state.stalks and v in t.state.stalks for u,v in t.state.edges)
    json.dumps(reduced); json.dumps(t.get_state())
    rotary.pulse(source); json.dumps(rotary.read_state())


def test_high_gini_does_not_harvest(monkeypatch):
    monkeypatch.setattr(rotary, "extract_gini_trajectory", lambda _: .9)
    transducer = rotary.HybridConditionStateTransducer()
    transducer.state.last_energy = 1
    result = transducer.pulse_mid_activity_evaluation("pass")
    assert result["verdict"] == "OBSTRUCTED" and "geometry_harvest" not in result
    rotary.harvest_geometry_resolution.assert_not_called()


def test_governor_replaces_context_and_resets_flags():
    gov = kv.TopologicalKVGovernor()
    gov.project_tokens(["longtoken", "short"])
    gov.identify_h0_maintainers(threshold=0)
    gov.identify_h0_maintainers(threshold=1)
    assert not any(token.is_h0_maintainer for token in gov.token_map.values())
    gov.project_tokens(["new"])
    assert list(gov.token_map) == [0] and gov.token_map[0].token_text == "new"
    gov.project_tokens([])
    assert not gov.token_map and not gov.current_context and not gov.h0_maintainers


def test_bridge_uses_supported_governor_api():
    bridge = hardware.LlamaCppTopologicalBridge(hardware.LlamaCppBridgeConfig())
    result = bridge.poll_hardware_and_govern(5.9)
    assert isinstance(result["evicted"], list)
    assert result["kv_trim_applied"] is False
    assert bridge.apply_ternary_gguf_map({"layers": {}})["applied"] is False
    assert bridge.route_ternary_forward("prompt", {"layers": {}})["status"] == "unsupported"


@pytest.mark.parametrize("text,width", [("aa bb cc dd",5),("abcdefghijk",3),("",4),("word",1)])
def test_chunker_no_duplicates_or_oversized_chunks(text,width):
    bridge = hardware.LlamaCppTopologicalBridge(hardware.LlamaCppBridgeConfig())
    chunks = bridge.rlm_chunk_and_feed(text,width)
    assert all(len(chunk)<=width for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_all_fallback_tools_dispatch_with_shared_contract(monkeypatch):
    server = mcp.MinimalMCP()
    assert set(server.tools) == set(mcp.TOOLS)
    for name in server.tools:
        monkeypatch.setitem(mcp.TOOLS,name,lambda **_: {"test": True})
        response = server.handle_request({"id":1,"method":"tools/call","params":{"name":name}})
        assert json.loads(response["result"]["content"][0]["text"]) == {"test":True}
    assert server.handle_request({"method":"notifications/initialized"}) is None


def test_quantizer_rejects_non_synthetic_mcp_models():
    with pytest.raises(ValueError, match="only"):
        mcp._run_svd_quantize("real-checkpoint")


def test_restriction_matrix_alignment_and_svd_loading_selection(monkeypatch):
    laplacian, names, restrictions = quantizer.build_weight_sheaf({"layer":np.ones((33,33))})
    assert laplacian.shape == restrictions.shape == (len(names),len(names))
    monkeypatch.setattr(quantizer, "build_weight_sheaf", lambda _: (np.diag([1.,10.,2.]),["low","high","medium"],np.eye(3)))
    result = quantizer.sheaf_svd_quantize({"layer":np.ones((2,2))},preserve_ratio=.1)
    assert result["global_h0_nodes"] == ["high"]
    zero = quantizer.sheaf_svd_quantize({"layer":np.zeros((2,2))})
    assert zero["layers"]["layer"]["counts"]["0"] == 4


def test_worker_restores_exact_bytes_and_does_not_touch_other_files(tmp_path):
    source = b'def sample():\n    return ("CHAOS", 42)\n'
    target = tmp_path / "target.py"; target.write_bytes(source)
    other = tmp_path / "other.txt"; other.write_bytes(b"keep)")
    originals = night.chaos_monkey_intent_injector(tmp_path)
    night.dispatch_sheaf_guardian(tmp_path,originals=originals)
    assert target.read_bytes() == source and other.read_bytes() == b"keep)"
    assert not hasattr(night,"transducer") and not hasattr(night,"kv_governor")


def test_worktree_failure_is_propagated(monkeypatch,tmp_path):
    run = Mock(side_effect=subprocess.CalledProcessError(1,["git"]))
    monkeypatch.setattr(night.subprocess,"run",run)
    with pytest.raises(subprocess.CalledProcessError): night.create_worktree(tmp_path,tmp_path/"missing")
    assert not (tmp_path/"missing").exists()
    assert isinstance(run.call_args.args[0],list)


def test_harvest_counter_and_path_scrubbing():
    assert oracle.harvest_geometry_resolution({}, {"verdict":"OBSTRUCTED"},1,[])["action"] != "geometry_harvested"
    def harvest(_):
        return oracle.harvest_geometry_resolution({}, {"verdict":"CONSISTENT","target":r"C:\host\worktrees\fiber\target.py"},1,[])
    with ThreadPoolExecutor(max_workers=4) as pool: receipts = list(pool.map(harvest,range(8)))
    assert sorted(item["stats"]["total_harvested"] for item in receipts) == list(range(1,9))
    records = [json.loads(line) for line in Path("shape_pairs.jsonl").read_text().splitlines()]
    assert len(records)==8 and all("C:" not in record["post_state"]["target"] for record in records)


def test_review_reports_only_checked_evidence():
    result = reviewer.review_artifact("def outer():\n inner()\n")
    rendered = reviewer.format_review(result)
    assert "not_checked" in rendered and "Verified (" not in rendered
    assert result["original_stalk_count"] > result["pruned_stalk_count"]
    json.dumps(result)


def test_hook_validates_complete_prospective_edit(tmp_path):
    file=tmp_path/"target.py";file.write_text("before=1\n")
    result=proposed_artifact({"tool_name":"Edit","tool_input":{"file_path":str(file),"old_string":"1","new_string":"2"}})
    assert result=="before=2\n" and file.read_text()=="before=1\n"
    assert proposed_artifact({"tool_name":"Read"}) is None


def test_fastmcp_registers_the_shared_surface(monkeypatch):
    registered = {}
    class FakeFastMCP:
        def __init__(self, name): assert name == "sheaf-condition-mcp"
        def tool(self, name):
            def register(function):
                registered[name] = function
                return function
            return register
        def run(self): pass
    monkeypatch.setattr(mcp, "FastMCP", FakeFastMCP)
    monkeypatch.delenv("SHEAF_MINIMAL_MCP", raising=False)
    mcp.main()
    assert registered == mcp.TOOLS


def test_real_minimal_tool_call_is_json_safe():
    response = mcp.MinimalMCP().handle_request({"id":1,"method":"tools/call","params":{"name":"rotate_condition","arguments":{"artifact":"def alpha():\n beta()\n"}}})
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["energy"] > 0 and result["verdict"] == "OBSTRUCTED"
    assert isinstance(result["hot_linkages"],list)


def test_static_catalog_has_unique_ids_and_actual_ratio():
    import llm_component_topological_analyzer as analyzer
    stalks, edges, info = analyzer.build_component_manifold()
    assert len(stalks) == len(analyzer.LLM_COMPONENTS)
    assert info["pruned_ratio"] == len(edges)/len(analyzer.LLM_INTEGRATION_LINKAGES)
    coherence = analyzer.compute_component_coherence(stalks,[("a","b","Berkouk-Ginot invariant")])
    assert coherence["hot_linkages"] == []


def test_hook_passes_or_blocks_with_machine_readable_input(tmp_path):
    import os
    hook=Path(__file__).resolve().parents[1]/"scripts/tool_use_hook.py"
    allowed={"tool_name":"Write","tool_input":{"file_path":str(tmp_path/"new.py"),"content":"pass\n"}}
    blocked={"tool_name":"Write","tool_input":{"file_path":str(tmp_path/"new.py"),"content":"def alpha():\n beta()\n"}}
    for event, code in [(allowed,0),(blocked,2)]:
        result=subprocess.run([sys.executable,str(hook)],input=json.dumps(event),text=True,capture_output=True,env=os.environ.copy())
        assert result.returncode==code, result.stderr
        assert result.stdout==""
