#!/usr/bin/env python3
"""LLMdex — Extract model metadata and tensor inventory from Ollama models."""

import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OLLAMA_MODELS = Path.home() / ".ollama" / "models"

# --- Safetensors component maps (Ollama v2 tensor-per-blob format) ---

ST_COMPONENT_MAP = {
    "self_attn.q_proj":       "attn.q",
    "self_attn.k_proj":       "attn.k",
    "self_attn.v_proj":       "attn.v",
    "self_attn.o_proj":       "attn.o",
    "self_attn.q_norm":       "norm.q",
    "self_attn.k_norm":       "norm.k",
    "linear_attn.in_proj_qkv": "linear_attn.qkv",
    "linear_attn.in_proj_z":  "linear_attn.z",
    "linear_attn.out_proj":   "linear_attn.o",
    "linear_attn.conv1d":     "linear_attn.conv",
    "linear_attn.A_log":      "linear_attn.A",
    "linear_attn.dt_bias":    "linear_attn.dt",
    "linear_attn.in_proj_a":  "linear_attn.a",
    "linear_attn.in_proj_b":  "linear_attn.b",
    "linear_attn.norm":       "linear_attn.norm",
    "mlp.gate":               "ff.router",
    "mlp.switch_mlp.gate_proj": "ff.expert.gate",
    "mlp.switch_mlp.up_proj":   "ff.expert.up",
    "mlp.switch_mlp.down_proj": "ff.expert.down",
    "mlp.shared_expert.gate_proj": "ff.shared.gate",
    "mlp.shared_expert.up_proj":   "ff.shared.up",
    "mlp.shared_expert.down_proj": "ff.shared.down",
    "mlp.shared_expert_gate":      "ff.shared_gate",
    "input_layernorm":        "norm.attn",
    "post_attention_layernorm": "norm.ffn",
}

ST_VISION_COMPONENT_MAP = {
    "attn.qkv":         "vt.attn.qkv",
    "attn.proj":        "vt.attn.proj",
    "mlp.linear_fc1":   "vt.ff.up",
    "mlp.linear_fc2":   "vt.ff.down",
    "norm1":            "vt.norm.attn",
    "norm2":            "vt.norm.ffn",
}

ST_GLOBAL_MAP = {
    "language_model.model.embed_tokens": "embed",
    "language_model.lm_head":            "output",
    "language_model.model.norm":         "norm.final",
    "vision_tower.patch_embed.proj":     "vt.patch_embed",
    "vision_tower.pos_embed":            "vt.pos_embed",
    "vision_tower.merger.linear_fc1":    "vt.merger.up",
    "vision_tower.merger.linear_fc2":    "vt.merger.down",
    "vision_tower.merger.norm":          "vt.merger.norm",
}

# --- GGUF component map ---

GGUF_COMPONENT_MAP = {
    "attn_q":              "attn.q",
    "attn_k":              "attn.k",
    "attn_v":              "attn.v",
    "attn_output":         "attn.o",
    "attn_qkv":            "attn.qkv",
    "attn_norm":           "norm.attn",
    "attn_q_norm":         "norm.q",
    "attn_k_norm":         "norm.k",
    "ffn_gate":            "ff.gate",
    "ffn_up":              "ff.up",
    "ffn_down":            "ff.down",
    "ffn_norm":            "norm.ffn",
    "ffn_gate_inp":        "ff.router",
    "ffn_gate_exps":       "ff.expert.gate",
    "ffn_up_exps":         "ff.expert.up",
    "ffn_down_exps":       "ff.expert.down",
    "post_attention_norm": "norm.post_attn",
    "post_ffw_norm":       "norm.post_ffn",

    # Mamba / SSM blocks (pure or hybrid linear-attention models)
    "ssm_in":              "linear_attn.in",
    "ssm_conv1d":          "linear_attn.conv",
    "ssm_x":               "linear_attn.x",
    "ssm_dt":              "linear_attn.dt",
    "ssm_dt_norm":         "linear_attn.dt_norm",
    "ssm_a":               "linear_attn.A",
    "ssm_b_norm":          "linear_attn.b_norm",
    "ssm_c_norm":          "linear_attn.c_norm",
    "ssm_d":               "linear_attn.d",
    "ssm_norm":            "linear_attn.norm",
    "ssm_out":             "linear_attn.o",
}

GGUF_GLOBAL_MAP = {
    "token_embd":  "embed",
    "output":      "output",
    "output_norm": "norm.final",
}

GGUF_SKIP = {"rope_freqs", "rope_factors_long", "rope_factors_short"}


# --- Common utilities ---

def resolve_blob(digest: str) -> Path:
    return OLLAMA_MODELS / "blobs" / digest.replace(":", "-")


def resolve_ollama_manifest(name: str, tag: str = "latest") -> dict:
    if "/" in name:
        namespace, model_name = name.split("/", 1)
    else:
        namespace, model_name = "library", name
    manifest_path = (
        OLLAMA_MODELS / "manifests" / "registry.ollama.ai" / namespace / model_name / tag
    )
    if not manifest_path.exists():
        sys.exit(f"Model not found: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def load_params(manifest: dict) -> dict | None:
    for layer in manifest["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.params":
            blob_path = resolve_blob(layer["digest"])
            with open(blob_path) as f:
                return json.load(f)
    return None


def compute_file_size(manifest: dict) -> int:
    return sum(l["size"] for l in manifest["layers"])


def detect_format(manifest: dict) -> str:
    for layer in manifest["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return "gguf"
        if layer["mediaType"] == "application/vnd.ollama.image.tensor":
            return "safetensors"
    return "unknown"


def derive_quantization(tensors: list) -> str:
    dtype_params: dict[str, int] = {}
    for t in tensors:
        if t["kind"] != "weight":
            continue
        dtype_params[t["dtype"]] = dtype_params.get(t["dtype"], 0) + t["n_params"]
    if not dtype_params:
        return "unknown"
    primary = max(dtype_params, key=dtype_params.get)
    others = sorted(dt for dt in dtype_params if dt != primary)
    if others:
        return f"{primary.lower()} ({', '.join(dt.lower() for dt in others)} embed/norms)"
    return primary.lower()


# --- MXFP8 dequantization ---

def _build_e4m3_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = (i >> 7) & 1
        exp = (i >> 3) & 0xF
        mantissa = i & 0x7
        if exp == 0:
            val = (mantissa / 8.0) * (2.0 ** -6)
        elif exp == 0xF and mantissa == 0x7:
            val = float('nan')
        else:
            val = (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7))
        lut[i] = -val if sign else val
    return lut

E4M3_LUT = _build_e4m3_lut()


def dequant_mxfp8(weight_u32: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    weight_bytes = weight_u32.view(np.uint8).reshape(
        weight_u32.shape[:-1] + (weight_u32.shape[-1] * 4,)
    )
    decoded = E4M3_LUT[weight_bytes]
    scales = np.ldexp(np.ones_like(scale_u8, dtype=np.float32), scale_u8.astype(np.int32) - 127)
    scales_expanded = np.repeat(scales, 32, axis=-1)
    return decoded * scales_expanded


# --- Tensor statistics ---

def compute_tensor_stats(data: np.ndarray) -> dict:
    flat = data.flatten().astype(np.float64)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"mean": 0, "std": 0, "rms": 0, "min": 0, "max": 0, "sparsity": 1.0}
    mean = float(finite.mean())
    std = float(finite.std())
    rms = float(np.sqrt(np.mean(finite ** 2)))
    return {
        "mean": round(mean, 8),
        "std": round(std, 6),
        "rms": round(rms, 6),
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "sparsity": round(float(np.mean(np.abs(finite) < 1e-6)), 6),
    }


# --- Safetensors extraction (Ollama v2 with per-tensor blobs) ---

def read_safetensors_header(blob_path: Path) -> dict:
    with open(blob_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def st_logical_shape(shape: list, dtype: str) -> list:
    if dtype == "U32" and len(shape) > 0:
        return shape[:-1] + [shape[-1] * 4]
    return shape


def st_logical_dtype(dtype: str, metadata: dict, tensor_name: str) -> str:
    if dtype == "U32":
        qt_key = f"{tensor_name}.quant_type"
        qt = metadata.get(qt_key, metadata.get("quant_type", "mxfp8"))
        return qt.upper()
    return dtype


def st_parse_name(full_name: str):
    suffix = full_name.rsplit(".", 1)[-1]
    kind = "bias" if suffix == "bias" else "weight"

    lm_layer = re.match(
        r"language_model\.model\.layers\.(\d+)\.(.+?)(?:\.(?:weight|bias))?$", full_name
    )
    if lm_layer:
        return int(lm_layer.group(1)), lm_layer.group(2), "language", kind

    vt_block = re.match(
        r"vision_tower\.blocks\.(\d+)\.(.+?)(?:\.(?:weight|bias))?$", full_name
    )
    if vt_block:
        return int(vt_block.group(1)), vt_block.group(2), "vision", kind

    for prefix, component in ST_GLOBAL_MAP.items():
        if full_name.startswith(prefix):
            domain = "vision" if component.startswith("vt.") else "language"
            return None, component, domain, kind

    return None, full_name, "unknown", kind


def st_map_component(raw_component: str, domain: str) -> str:
    if domain == "vision":
        return ST_VISION_COMPONENT_MAP.get(raw_component, f"vt.{raw_component}")
    if domain == "language":
        return ST_COMPONENT_MAP.get(raw_component, raw_component)
    return raw_component


ST_DTYPE_NP = {"BF16": ">u2", "F16": np.float16, "F32": np.float32}


def st_read_tensor_data(blob_path: Path, header: dict, key: str) -> np.ndarray | None:
    info = header[key]
    dtype_stored = info["dtype"]
    shape_stored = info["shape"]
    start, end = info["data_offsets"]

    with open(blob_path, "rb") as f:
        raw_header = f.read(8)
        header_size = struct.unpack("<Q", raw_header)[0]
        data_offset = 8 + header_size

    if dtype_stored == "U32":
        scale_key = f"{key}.scale"
        if scale_key not in header:
            return None
        scale_info = header[scale_key]
        s_start, s_end = scale_info["data_offsets"]
        with open(blob_path, "rb") as f:
            f.seek(data_offset + start)
            weight_raw = np.frombuffer(f.read(end - start), dtype=np.uint32).reshape(shape_stored)
            f.seek(data_offset + s_start)
            scale_raw = np.frombuffer(f.read(s_end - s_start), dtype=np.uint8).reshape(scale_info["shape"])
        return dequant_mxfp8(weight_raw, scale_raw)

    if dtype_stored == "BF16":
        with open(blob_path, "rb") as f:
            f.seek(data_offset + start)
            raw = np.frombuffer(f.read(end - start), dtype=np.uint16)
        return raw.astype(np.uint32) << 16  # BF16 → F32 via bit shift

    np_dtype = ST_DTYPE_NP.get(dtype_stored)
    if np_dtype is None:
        return None
    with open(blob_path, "rb") as f:
        f.seek(data_offset + start)
        return np.frombuffer(f.read(end - start), dtype=np_dtype)


def extract_safetensors(manifest: dict, with_stats: bool = True) -> list[dict]:
    tensors = []
    for layer in manifest["layers"]:
        if layer["mediaType"] != "application/vnd.ollama.image.tensor":
            continue

        blob_path = resolve_blob(layer["digest"])
        header = read_safetensors_header(blob_path)
        metadata = header.get("__metadata__", {})

        for key, info in header.items():
            if key == "__metadata__" or key.endswith(".scale"):
                continue

            shape_stored = info["shape"]
            dtype_stored = info["dtype"]
            shape_log = st_logical_shape(shape_stored, dtype_stored)
            dtype_log = st_logical_dtype(dtype_stored, metadata, key)
            n_params = 1
            for d in shape_log:
                n_params *= d

            layer_idx, raw_comp, domain, kind = st_parse_name(key)
            component = st_map_component(raw_comp, domain)

            entry = {
                "name": key,
                "layer": layer_idx,
                "component": component,
                "domain": domain,
                "kind": kind,
                "shape": shape_log,
                "dtype": dtype_log,
                "n_params": n_params,
            }

            if with_stats:
                data = st_read_tensor_data(blob_path, header, key)
                if data is not None:
                    if data.dtype == np.uint32:
                        data = data.view(np.float32)
                    entry["stats"] = compute_tensor_stats(data)
                    del data

            tensors.append(entry)

    return tensors


def load_st_config(manifest: dict) -> dict | None:
    for layer in manifest["layers"]:
        if layer.get("name") == "config.json":
            blob_path = resolve_blob(layer["digest"])
            with open(blob_path) as f:
                return json.load(f)
    return None


def build_st_meta(name: str, config: dict | None, tensors: list, manifest: dict) -> dict:
    tc = config.get("text_config", {}) if config else {}
    vc = config.get("vision_config", {}) if config else {}

    params_total = sum(t["n_params"] for t in tensors)
    params_vision = sum(
        t["n_params"] for t in tensors if t["component"].startswith("vt.")
    )
    params_language = params_total - params_vision

    n_experts = tc.get("num_experts", 0)
    n_experts_active = tc.get("num_experts_per_tok", 0)
    expert_params = sum(
        t["n_params"] for t in tensors if t["component"].startswith("ff.expert.")
    )
    if n_experts > 0 and n_experts_active > 0 and expert_params > 0:
        params_active = params_language - expert_params + (
            expert_params * n_experts_active // n_experts
        )
    else:
        params_active = params_language

    layer_types = tc.get("layer_types")
    full_attn_layers = None
    if layer_types:
        full_attn_layers = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

    meta = {
        "name": name,
        "format": "safetensors",
        "architecture": config.get("model_type", "unknown") if config else "unknown",
        "params_total": params_total,
        "params_language": params_language,
        "params_vision": params_vision,
        "params_active": params_active,
        "quantization": derive_quantization(tensors),
        "file_size": compute_file_size(manifest),
        "n_layers": tc.get("num_hidden_layers"),
        "n_heads": tc.get("num_attention_heads"),
        "n_kv_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "d_embed": tc.get("hidden_size"),
        "d_ff": tc.get("intermediate_size"),
        "d_ff_expert": tc.get("moe_intermediate_size"),
        "d_ff_shared": tc.get("shared_expert_intermediate_size"),
        "context_length": tc.get("max_position_embeddings"),
        "vocab_size": tc.get("vocab_size"),
        "n_experts": n_experts if n_experts > 0 else None,
        "n_experts_active": n_experts_active if n_experts_active > 0 else None,
        "rope_theta": tc.get("rope_parameters", {}).get("rope_theta"),
        "layer_types": layer_types,
        "full_attn_layers": full_attn_layers,
        "linear_attn": {
            "num_key_heads": tc.get("linear_num_key_heads"),
            "num_value_heads": tc.get("linear_num_value_heads"),
            "key_head_dim": tc.get("linear_key_head_dim"),
            "value_head_dim": tc.get("linear_value_head_dim"),
            "conv_kernel": tc.get("linear_conv_kernel_dim"),
        } if tc.get("linear_num_key_heads") else None,
    }

    if vc:
        meta["vision"] = {
            "depth": vc.get("depth"),
            "hidden_size": vc.get("hidden_size"),
            "num_heads": vc.get("num_heads"),
            "intermediate_size": vc.get("intermediate_size"),
            "patch_size": vc.get("patch_size"),
            "out_hidden_size": vc.get("out_hidden_size"),
        }

    params_blob = load_params(manifest)
    if params_blob:
        meta["runtime_params"] = params_blob

    return meta


# --- GGUF extraction ---

def gguf_read_field(field) -> any:
    import numpy as np
    if len(field.data) == 0:
        return None
    if len(field.data) == 1:
        val = field.parts[field.data[0]]
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            if val.dtype.kind in ('U', 'S'):
                return str(val[0]) if val.size == 1 else [str(v) for v in val]
            if val.dtype == np.uint8:
                return val.tobytes().decode('utf-8', errors='replace')
            return val.tolist()
        return val
    vals = []
    for d in field.data:
        v = field.parts[d]
        if isinstance(v, np.ndarray):
            if v.size == 1:
                vals.append(v.item())
            elif v.dtype == np.uint8:
                vals.append(v.tobytes().decode('utf-8', errors='replace'))
            else:
                vals.append(v.tolist())
        else:
            vals.append(v)
    if len(vals) == 1:
        return vals[0]
    if all(isinstance(v, int) and 0 < v < 128 for v in vals):
        return bytes(vals).decode('utf-8', errors='replace')
    return vals


def gguf_parse_tensor_name(name: str):
    suffix = name.rsplit(".", 1)[-1]
    kind = "bias" if suffix == "bias" else "weight"

    blk = re.match(r"blk\.(\d+)\.(.+?)(?:\.(?:weight|bias))?$", name)
    if blk:
        return int(blk.group(1)), blk.group(2), kind

    base = re.sub(r"\.(?:weight|bias)$", "", name)
    return None, base, kind


GGUF_FLOAT_TYPES = {"F16", "F32", "BF16"}


def gguf_dequant_tensor(tensor) -> np.ndarray:
    from gguf import dequantize as gguf_dequantize
    dtype_name = tensor.tensor_type.name
    if dtype_name in GGUF_FLOAT_TYPES:
        if dtype_name == "F16":
            return tensor.data.astype(np.float32) if tensor.data.dtype == np.float16 else tensor.data.view(np.float16).astype(np.float32)
        if dtype_name == "BF16":
            raw = tensor.data.view(np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return tensor.data.view(np.float32) if tensor.data.dtype != np.float32 else tensor.data
    return gguf_dequantize(tensor.data, tensor.tensor_type)


def extract_gguf(manifest: dict, with_stats: bool = True) -> tuple[list[dict], dict]:
    from gguf import GGUFReader

    gguf_blob = None
    for layer in manifest["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            gguf_blob = layer
            break
    if not gguf_blob:
        return [], {}

    blob_path = resolve_blob(gguf_blob["digest"])
    reader = GGUFReader(str(blob_path))

    metadata = {}
    for key, field in reader.fields.items():
        if "tokenizer" in key or "token" in key:
            continue
        metadata[key] = gguf_read_field(field)

    tensors = []
    for t in reader.tensors:
        base_name = re.sub(r"\.(?:weight|bias)$", "", t.name)
        if base_name in GGUF_SKIP:
            continue

        shape_gguf = [int(d) for d in t.shape]
        shape_rowmajor = list(reversed(shape_gguf))
        dtype = t.tensor_type.name

        n_params = 1
        for d in shape_rowmajor:
            n_params *= d

        layer_idx, raw_comp, kind = gguf_parse_tensor_name(t.name)

        if layer_idx is not None:
            component = GGUF_COMPONENT_MAP.get(raw_comp, raw_comp)
        else:
            component = GGUF_GLOBAL_MAP.get(raw_comp, raw_comp)

        entry = {
            "name": t.name,
            "layer": layer_idx,
            "component": component,
            "domain": "language",
            "kind": kind,
            "shape": shape_rowmajor,
            "dtype": dtype,
            "n_params": n_params,
        }

        if with_stats:
            data = gguf_dequant_tensor(t)
            entry["stats"] = compute_tensor_stats(data)
            del data

        tensors.append(entry)

    return tensors, metadata


def build_gguf_meta(name: str, tensors: list, gguf_meta: dict, manifest: dict) -> dict:
    arch = gguf_meta.get("general.architecture", "unknown")
    arch_prefix = f"{arch}."

    def get(key, default=None):
        return gguf_meta.get(f"{arch_prefix}{key}", gguf_meta.get(key, default))

    params_total = sum(t["n_params"] for t in tensors)

    n_experts = get("expert_count", 0)
    n_experts_active = get("expert_used_count", 0)
    expert_params = sum(
        t["n_params"] for t in tensors if t["component"].startswith("ff.expert.")
    )
    if n_experts and n_experts_active and expert_params > 0:
        params_active = params_total - expert_params + (
            expert_params * n_experts_active // n_experts
        )
    else:
        params_active = params_total

    n_heads = get("attention.head_count")
    n_kv_heads = get("attention.head_count_kv")
    d_embed = get("embedding_length")
    head_dim = d_embed // n_heads if (d_embed and n_heads) else None

    meta = {
        "name": name,
        "format": "gguf",
        "architecture": arch,
        "params_total": params_total,
        "params_language": params_total,
        "params_vision": 0,
        "params_active": params_active,
        "quantization": derive_quantization(tensors),
        "file_size": compute_file_size(manifest),
        "n_layers": get("block_count"),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "d_embed": d_embed,
        "d_ff": get("feed_forward_length"),
        "d_ff_expert": None,
        "d_ff_shared": None,
        "context_length": get("context_length"),
        "vocab_size": get("vocab_size"),
        "n_experts": n_experts if n_experts else None,
        "n_experts_active": n_experts_active if n_experts_active else None,
        "rope_theta": get("rope.freq_base"),
        "layer_types": None,
        "full_attn_layers": None,
        "linear_attn": None,
    }

    params_blob = load_params(manifest)
    if params_blob:
        meta["runtime_params"] = params_blob

    return meta


# --- Main pipeline ---

def extract_model(name: str, tag: str = "latest") -> dict:
    manifest = resolve_ollama_manifest(name, tag)
    fmt = detect_format(manifest)

    if fmt == "safetensors":
        config = load_st_config(manifest)
        tensors = extract_safetensors(manifest)
        meta = build_st_meta(name, config, tensors, manifest)
    elif fmt == "gguf":
        tensors, gguf_meta = extract_gguf(manifest)
        meta = build_gguf_meta(name, tensors, gguf_meta, manifest)
    else:
        sys.exit(f"Unknown model format for {name}")

    tensor_stats = []
    for t in tensors:
        entry = {
            "name": t["name"],
            "layer": t["layer"],
            "component": t["component"],
            "domain": t["domain"],
            "kind": t["kind"],
            "shape": t["shape"],
            "dtype": t["dtype"],
            "n_params": t["n_params"],
        }
        if "stats" in t:
            entry.update(t["stats"])
        tensor_stats.append(entry)

    return {"meta": meta, "tensor_stats": tensor_stats}


def list_ollama_models() -> list[str]:
    manifests_dir = OLLAMA_MODELS / "manifests"
    if not manifests_dir.exists():
        return []
    models = []
    for manifest_file in sorted(manifests_dir.rglob("*")):
        if not manifest_file.is_file():
            continue
        rel = manifest_file.relative_to(manifests_dir)
        parts = rel.parts
        if len(parts) < 4:
            continue
        registry, namespace, model_name, tag = parts[0], parts[1], parts[2], parts[3]
        if namespace == "library":
            display = f"{model_name}:{tag}" if tag != "latest" else model_name
        else:
            display = f"{namespace}/{model_name}:{tag}" if tag != "latest" else f"{namespace}/{model_name}"
        models.append(display)
    return models


def main():
    parser = argparse.ArgumentParser(description="LLMdex — Extract model metadata")
    parser.add_argument("model", nargs="?", help="Ollama model name (e.g. qwen3.6-coding)")
    parser.add_argument("--tag", default="latest", help="Model tag (default: latest)")
    parser.add_argument("--all", action="store_true", help="Extract all installed Ollama models")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = list_ollama_models()
        if not models:
            sys.exit("No Ollama models found.")
        print(f"Found {len(models)} model(s)")
    else:
        models = [f"{args.model}:{args.tag}" if args.tag != "latest" else args.model]

    for model_spec in models:
        if ":" in model_spec:
            name, tag = model_spec.rsplit(":", 1)
        else:
            name, tag = model_spec, "latest"

        print(f"\nExtracting {name}:{tag}...")
        try:
            result = extract_model(name, tag)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        meta = result["meta"]
        tensors = result["tensor_stats"]
        p_total = meta["params_total"]
        p_active = meta["params_active"]
        has_stats = any("rms" in t for t in tensors)

        print(f"  Format: {meta['format']} | Architecture: {meta['architecture']}")
        print(f"  Layers: {meta['n_layers']}, d_embed: {meta['d_embed']}, d_ff: {meta.get('d_ff')}")
        print(f"  Params total: {p_total:,} ({p_total / 1e9:.1f}B)")
        if p_active != p_total:
            print(f"  Params active: {p_active:,} ({p_active / 1e9:.1f}B)")
        print(f"  Stats: {'computed' if has_stats else 'metadata only'}")
        print(f"  Quantization: {meta['quantization']}")
        print(f"  Tensors: {len(tensors)}")
        print(f"  File size: {meta['file_size'] / 1e9:.1f} GB")

        safe_name = name.replace("/", "_").replace(":", "_")
        out_path = output_dir / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
