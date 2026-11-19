#!/usr/bin/env python3
"""Adapt Write/Edit hook input to a prospective artifact and the simulation gate."""
import json
from pathlib import Path
import sys


def proposed_artifact(event):
    tool = event.get("tool_name")
    if tool not in {"Write", "Edit"}: return None
    params = event.get("tool_input", {})
    path = Path(params["file_path"])
    if tool == "Write": return params["content"]
    original = path.read_text(encoding="utf-8")
    old, new = params["old_string"], params["new_string"]
    if not old or old not in original: raise ValueError("Edit old_string was not found")
    if not params.get("replace_all") and original.count(old) != 1:
        raise ValueError("Edit old_string is ambiguous")
    return original.replace(old, new, -1 if params.get("replace_all") else 1)


def main():
    post = len(sys.argv) > 1 and sys.argv[1] == "--post"
    try:
        if len(sys.argv) > 1 and not post:
            path = Path(sys.argv[1])
            artifact = sys.argv[2] if len(sys.argv) > 2 else path.read_text(encoding="utf-8")
        else:
            event = json.load(sys.stdin)
            if post:
                params = event.get("tool_input", {})
                path = Path(params.get("file_path", ""))
                if event.get("tool_name") not in {"Write", "Edit"} or not path.is_file(): return 0
                artifact = path.read_text(encoding="utf-8")
            else:
                artifact = proposed_artifact(event)
        if artifact is None: return 0
        from sheaf_coderabbit import review_artifact, format_review
        review = review_artifact(artifact, context="post-tool-review" if post else "pre-tool-review")
        if post:
            output = Path(__file__).resolve().parents[1] / "reports/current-task-yield.md"
            with output.open("a", encoding="utf-8") as stream:
                stream.write(format_review(review) + "\n")
            return 0
        if review["verdict"] != "CONSISTENT" or review["energy"] >= 1e-3 or review.get("gini_curve", 1) > 0.8:
            print(json.dumps({"reason": "Simulation gate obstructed", "review": review}), file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Sheaf hook configuration/input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
