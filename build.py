#!/usr/bin/env python3
"""LLMdex — Assemble extracted model data into the atlas HTML page."""

import json
import sys
from pathlib import Path


def main():
    data_dir = Path("data")
    template_path = Path("template.html")
    dist_dir = Path("dist")

    if not template_path.exists():
        sys.exit("template.html not found")

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        sys.exit("No JSON files in data/")

    models = []
    for f in json_files:
        models.append(json.loads(f.read_text()))
        print(f"  Loaded {f.name}")

    template = template_path.read_text()
    output = template.replace('"INJECT_DATA"', json.dumps(models))

    dist_dir.mkdir(exist_ok=True)
    out_path = dist_dir / "index.html"
    out_path.write_text(output)
    print(f"\n→ {out_path} ({len(models)} model(s), {out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
