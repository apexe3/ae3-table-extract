# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is free and open source: you may use, modify and distribute
# it, including in commercial products, under the terms of the License.
# AE3-TL is created and maintained by Team APEX:E3.

"""ONNX Runtime sessions for the AE3-TL network and its relation head.

One .onnx file runs on CPU and on GPU; the provider list decides which, not the file.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime


@dataclass(frozen=True)
class OnnxSettings:
    """Which files to run, on which provider, with how many threads."""

    network_file: Path
    relation_file: Path
    provider: str
    intra_op_threads: int
    device_id: int


def load_onnx_settings(exports_dir: Path) -> OnnxSettings:
    """Read the AE3TL_ONNX_* variables, defaulting to the files shipped in exports/."""
    provider = (os.environ.get("AE3_PROVIDER")
                or os.environ.get("AE3TL_ONNX_PROVIDER")
                or "auto")
    if provider not in ("auto", "cpu", "cuda", "tensorrt", "coreml"):
        raise ValueError(
            f"AE3_PROVIDER must be auto, cpu, cuda, tensorrt or coreml, got {provider!r}")
    return OnnxSettings(
        network_file=Path(os.environ.get("AE3TL_ONNX_FILE",
                                         exports_dir / "ae3tl_pages_v3_ep1_fp32_opset17.onnx")),
        relation_file=Path(os.environ.get("AE3TL_ONNX_RELATION_FILE",
                                          exports_dir / "ae3tl_relhead_v3_ep1_fp32_opset17.onnx")),
        provider=provider,
        intra_op_threads=int(os.environ.get("AE3TL_ONNX_THREADS", 0)),
        device_id=int(os.environ.get("AE3TL_ONNX_DEVICE_ID", 0)),
    )


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def session_options(intra_op_threads: int, provider: str) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    if _flag("AE3_DETERMINISTIC"):
        options.use_deterministic_compute = True
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options


def available_providers() -> list[str]:
    return list(onnxruntime.get_available_providers())


def resolve_provider(provider: str) -> str:
    """Turn ``auto`` into the best provider this machine actually has."""
    if provider not in ("auto", ""):
        return provider
    have = set(available_providers())
    if "TensorrtExecutionProvider" in have:
        return "tensorrt"
    if "CUDAExecutionProvider" in have:
        return "cuda"
    if "CoreMLExecutionProvider" in have:
        return "coreml"
    return "cpu"


def coreml_options() -> dict:
    """Options for the CoreML provider; the cache stops it recompiling on every start."""
    cache = os.environ.get("AE3_COREML_CACHE") or str(Path.home() / ".cache" / "ae3tl-coreml")
    return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1", "ModelCacheDirectory": cache}


def cuda_options(device_id: int) -> dict:
    """CUDA provider options.

    use_tf32 is off by default here: TF32 shortens the mantissa, and a batch of one
    takes a different kernel path than a batch of many, so results move with the batch.
    AE3_TF32=1 turns it back on for speed.
    """
    options = {"device_id": device_id}
    if _flag("AE3_DETERMINISTIC"):
        options["cudnn_conv_algo_search"] = "HEURISTIC"
    tf32 = os.environ.get("AE3_TF32")
    if tf32 is not None and tf32.strip() != "":
        options["use_tf32"] = "1" if tf32.strip().lower() in ("1", "true", "yes", "on") else "0"
    return options


def providers(provider: str, device_id: int) -> list:
    """The provider list for one runtime package.

    TensorRT is not in v0.1: it needs a separate library and a first-run engine build. The option
    exists so the runtime package can be swapped without touching the service.
    """
    provider = resolve_provider(provider)
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "coreml":
        return [("CoreMLExecutionProvider", coreml_options()), "CPUExecutionProvider"]
    if provider == "cuda":
        return [("CUDAExecutionProvider", cuda_options(device_id)), "CPUExecutionProvider"]
    return [("TensorrtExecutionProvider", {"device_id": device_id,
                                           "trt_engine_cache_enable": True}),
            ("CUDAExecutionProvider", cuda_options(device_id)), "CPUExecutionProvider"]


def make_session(path: Path, settings: OnnxSettings, label: str) -> "onnxruntime.InferenceSession":
    """Create one session on the chosen provider, or stop with the reason.

    On a Mac the CoreML provider is required: if it cannot start, the error names the fix instead of
    silently running on the CPU.
    """
    chosen = resolve_provider(settings.provider)
    options = session_options(settings.intra_op_threads, chosen)
    model = path
    if chosen == "coreml":
        if "CoreMLExecutionProvider" not in available_providers():
            raise RuntimeError(
                "CoreMLExecutionProvider is not available in this onnxruntime build. On Apple Silicon "
                "install the standard wheel: pip install -U 'onnxruntime>=1.20'. Available providers: "
                + ", ".join(available_providers()))
        static = path.with_name(f"{path.stem}_b1{path.suffix}")
        if static.exists():
            model = static
        else:
            try:
                options.add_free_dimension_override_by_name("batch", 1)
            except Exception as exc:
                raise RuntimeError(
                    "This onnxruntime build has no free-dimension override; install "
                    "'onnxruntime>=1.20'.") from exc
    try:
        session = onnxruntime.InferenceSession(
            str(model), options, providers=providers(chosen, settings.device_id))
    except Exception as exc:
        raise RuntimeError(
            f"{label}: the {chosen} provider failed to start: {type(exc).__name__}: {exc}. "
            f"Model: {model.name}. Providers: {', '.join(available_providers())}") from exc
    print(f"[ae3-tl] {label}: provider {chosen}; model {model.name}", file=sys.stderr)
    return session


class OnnxNetwork:
    """The detector: one page batch in, logits, boxes, query embeddings and encoder indices out."""

    def __init__(self, settings: OnnxSettings):
        self.settings = settings
        self.session = make_session(settings.network_file, settings, "page model")
        self.input_name = self.session.get_inputs()[0].name

    def run(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logits, boxes, embeddings, indices = self.session.run(
            None, {self.input_name: np.ascontiguousarray(images, dtype=np.float32)})
        return logits, boxes, embeddings, indices


class OnnxRelationHead:
    """Caption or footer query embeddings with table query embeddings -> one edge logit per pair."""

    def __init__(self, settings: OnnxSettings):
        self.session = make_session(settings.relation_file, settings, "relation head")

    def run(self, embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        logits = self.session.run(
            None, {"query_embeddings": np.ascontiguousarray(embeddings, dtype=np.float32),
                   "pairs": np.ascontiguousarray(pairs, dtype=np.int64)})[0]
        return logits
