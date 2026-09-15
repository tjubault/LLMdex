#!/usr/bin/env python3
"""LLMdex — build your own atlas from your local Ollama models, one command.

Equivalent to running, in order:

    python extract.py --all
    python build.py
    open dist/index.html

but as a single script, with a summary of anything worth knowing afterward
(e.g. tensors from an architecture this project hasn't seen before).
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

import extract


def main():
    models = extract.list_ollama_models()
    if not models:
        sys.exit(
            "No Ollama models found in ~/.ollama/models — pull one first, "
            "e.g. `ollama pull qwen2.5:7b`."
        )

    print(f"Found {len(models)} Ollama model(s): {', '.join(models)}\n")

    warnings = extract.extract_models(models, Path("data"))

    print("\nAssembling the atlas...")
    subprocess.run([sys.executable, "build.py"], check=True)

    if warnings:
        print(
            "\nHeads up: some tensors weren't recognized and will show up "
            "lumped in with feed-forward in the spectrum (probably an "
            "architecture this project hasn't been taught yet — the atlas "
            "still works, just less precisely for these models):"
        )
        for name, comps in warnings.items():
            preview = ", ".join(comps[:6]) + ("…" if len(comps) > 6 else "")
            print(f"  - {name}: {preview}")

    out_path = Path("dist/index.html").resolve()
    print(f"\nAtlas ready: {out_path}")
    webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
