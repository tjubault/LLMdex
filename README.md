# LLMdex

A visual atlas for local LLMs — a field guide for the models sitting in your Ollama library. Every model gets its own specimen plate: architecture at a glance, weight statistics, and a true-to-scale comparison against every other model in the collection. Styled like a naturalist's herbarium, not a dashboard.

**Live atlas:** https://tjubault.github.io/LLMdex/

## What's on a plate

- **Identity & stats** — parameters, layers, context length, vocabulary, quantization, size on disk.
- **Architectural mini-specimen** — attention vs. feed-forward drawn to scale, GQA boundary, MoE striping, vision tower (when present).
- **Layer morphology** — every weight tensor of a representative layer, drawn proportional to its real shape.
- **Weight spectrum** — a layer × component heatmap of tensor magnitudes (RMS), normalized per component family so the contrast stays meaningful.
- **Comparative plate** — every model in the collection, side by side, true to scale by size on disk.

The whole page is a single self-contained HTML file: vanilla JS/CSS/SVG, zero external dependencies, dark/light theme, FR/EN toggle.

## How it works

```
GGUF or safetensors → extract.py → JSON (data/) → build.py → dist/index.html
```

Three independent pieces connected by a JSON schema in between:

- `extract.py` reads one model (GGUF via `GGUFReader`, or safetensors via its config) and writes `data/<model>.json` — architecture metadata plus per-tensor statistics.
- `build.py` reads every JSON in `data/`, injects it into `template.html`, and writes `dist/index.html` (mirrored to `docs/` for GitHub Pages).
- `template.html` is the atlas page itself.

## Quick start

Build an atlas of your own local Ollama models, one command:

```bash
pip install gguf numpy
python make_atlas.py
```

This extracts every model installed in `~/.ollama/models`, assembles the atlas, and opens it in your browser. It's just a wrapper — equivalent to running the three steps below yourself, which is also how you extract a single model instead of everything:

```bash
python extract.py <model-name>     # one Ollama model, e.g. qwen2.5:7b
python extract.py --all            # every model installed locally
python build.py                    # → dist/index.html
open dist/index.html
```

`extract.py` reads GGUF tensors directly (via `GGUFReader`) and Ollama's safetensors-per-blob format (via each blob's config). If it runs into tensor names it doesn't recognize for a given architecture, it'll still produce a plate — those tensors just get grouped under feed-forward in the spectrum by default, and both `extract.py` and `make_atlas.py` print a warning naming them so you know the plate is approximate for that model.

## Roadmap

- [ ] More in-depth explanations throughout the plates
- [ ] A section showing how information actually flows through a model (the forward pass), not just its static weight structure
- [ ] A more "artistic" rendering style, as an alternative to the current technical/naturalist look
