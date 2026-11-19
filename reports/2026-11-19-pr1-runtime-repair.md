# PR 1 runtime repair — 2026-11-19

Reviewed original head `85191c651c436cd6f349bc01e4132a53e16eb102` on `phase-0`.
Merged actual git base `e5b36fe00f0a8ab73da84c9d6cf381135d32f59b` from `main`,
retaining its PolyForm license and NOTICE. The PR connector had reported an
older base SHA; git transport supplied the current base.

## Repairs

- Fixed invalid MCP Python syntax and shared tool registration/dispatch across
  FastMCP and the minimal stdio server. Stdio diagnostics go to stderr; tool
  responses use MCP content blocks and structured errors.
- Used the governor's implemented API; rebuilt token maps and maintainer flags
  per context; fixed duplicated/oversized chunks.
- Connected AST edges to the actual stalk IDs, restored nested caller context,
  honored the pruning flag, and exported JSON-safe linkage records.
- Replaced destructive graph matching and fabricated boundaries with conservative
  free-face graph collapses and oriented incidence matrices. The energy gate
  retains the original dependency graph because graph homology alone does not
  establish preservation of sheaf energy. Count/ratio reports use actual inputs.
- Replaced pulse-count-based coherence shifts with measured energy differences;
  obstructed verdicts cannot produce harvest receipts. Harvest counts reflect
  successful append records under a lock, with portable worktree-relative paths.
- Gave each daemon worker its own transducer/governor; aggregate results on the
  orchestrator thread; propagate git worktree failures; restore exact recorded
  bytes instead of stripping arbitrary text or parentheses.
- Implemented a checked CPU F32 C++ reference forward with correct ggml dimensions,
  signed weights, code validation, destination writes, and explicit status codes.
  The unimplemented JSON loader initializes every output and returns unsupported.
  Bridge APIs likewise report unsupported/unavailable instead of claiming maps
  were applied or physical KV trimming occurred.
- Aligned quantizer matrix dimensions, selected representative nodes using retained
  singular-subspace loadings, accepted checkpoint `--layers all`, and removed
  pickle checkpoint loading and silent checkpoint-to-synthetic fallback. Dense
  demos are bounded to 2048 feature nodes and use small defaults.
- Removed duplicate component IDs and arbitrary half-edge pruning, used measured
  ratios and word-boundary flags, and reject the unsupported `--archive` option.
  Regenerated the static catalog with explicit proxy limitations.
- Fixed bootstrap branch synchronization; register both stdin-JSON Write/Edit
  hooks in project settings. Hooks use Python without jq/bc; bootstrap checks
  prerequisites before installing and preserves global user settings.
- Pinned the GitHub MCP package to registry version 2025.4.8 and declared stdio;
  clarified generated-map policy, added valid skill frontmatter, fixed Markdown
  defects, removed duplicate roadmap text, and removed unconditional proof claims
  from generated review reports.

## Verification actually performed

- `python -m pytest -q tests`: 27 passed.
- `python scripts/test_homological_smoke.py`: passed synthetic graph Betti,
  boundary-squared-zero, and original-graph obstruction checks.
- `python -W error -m compileall -q -f scripts`: passed.
- `bash scripts/validate-sheaf.sh`: passed the repository validation helper.
- `bash -n hooks/pre-tool-use.sh scripts/bootstrap-harness.sh`: passed.
- Stdio subprocess integration: initialization, seven-tool listing, nonzero AST
  obstruction, current-state prune dispatch, and tiny synthetic quantization
  all returned parseable JSON without stdout diagnostic contamination.
- C++ reference test compiled with MSVC 14.44 using `/std:c++17 /W4 /WX` and ran
  successfully against a minimal test-only ggml tensor header. It checks signed
  forward results, maintainer behavior, invalid codes/counts, and loader outputs.
- `git diff --check`: passed.

Tests used isolated dependencies and temporary/output directories on D:. No
hardware daemon, bootstrap install, real model, or external dataset was run.

## Limits

These checks verify runtime contracts and small synthetic graph cases. They do
not verify a sheaf preservation theorem, persistent Betti/Gini signatures, model
quality, accelerator behavior, or a 6GB memory bound. The custom llama.cpp loader,
CTNN attention, physical KV trimming, and actual QLoRA training remain unimplemented.
Their interfaces now expose that limitation instead of emitting success claims.
The PR's existing non-draft state is retained; no merge or review comments are
part of this repair.
