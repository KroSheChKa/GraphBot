"""
Local sandbox for small test scripts.

Use this file for quick experiments so the project root stays clean.
Save results under ../outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def save_sample_json(name: str = "sample.json") -> Path:
    """Example: write a test JSON file to outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps({"ok": True, "source": "sandbox/tester_programs.py"}, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = save_sample_json()
    print(f"Wrote {out}")
