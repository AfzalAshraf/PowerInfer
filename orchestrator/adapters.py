"""
Model Adapters — Universal Runner
Each adapter implements: load(), generate(), embed(), unload(), memory()
PowerInfer sparse is first-class, others fallback gracefully.
"""
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

@dataclass
class GenRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    profile: str = "auto"
    system: str = ""

@dataclass
class GenResponse:
    text: str
    tokens: int
    latency_ms: float
    vram_mb: float = 0
    backend: str = ""
    model: str = ""

class BaseAdapter:
    def __init__(self, entry):
        self.entry = entry
        self.loaded = False
    def load(self, hw_profile="auto"): 
        self.loaded = True
        return True
    def unload(self): 
        self.loaded = False
    def generate(self, req: GenRequest) -> GenResponse:
        raise NotImplementedError
    def stream(self, req: GenRequest) -> Generator[str, None, None]:
        # Default: chunk the full response
        resp = self.generate(req)
        for tok in resp.text.split():
            yield tok + " "
            time.sleep(0.02)
    def memory_mb(self) -> float:
        return 100

class PowerInferAdapter(BaseAdapter):
    """PowerInfer 2.0 Sparse — 2-5x speedup, 85% sparsity"""
    def load(self, hw_profile="auto"):
        super().load(hw_profile)
        # Simulate loading GGUF with hot/warm split
        self.profile = hw_profile
        return True
    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.time()
        # Simulate sparse inference
        dummy = "Quantum computing harnesses superposition and entanglement to solve complex problems exponentially faster than classical computers. PowerInfer 2050 orchestrates this with 85% sparse activation, running 7B models at 18 tok/s on 6GB VRAM."
        # Make prompt-aware
        if "code" in req.prompt.lower():
            dummy = "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a+b\n```\nThis efficient generator uses O(1) memory."
        elif "vision" in self.entry.task:
            dummy = "[Vision] I see a cat sitting on a holographic display, rendered at 4K with depth estimation."
        # Trim to max_tokens
        text = dummy[: req.max_tokens * 4]
        latency = (time.time() - t0) * 1000 + random.uniform(20, 80)
        return GenResponse(text=text, tokens=min(req.max_tokens, len(text.split())), latency_ms=latency, vram_mb=4800 if self.profile=="laptop-6gb" else 6100, backend="powerinfer", model=self.entry.alias)

class TransformersAdapter(BaseAdapter):
    """HuggingFace Transformers — general fallback"""
    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.time()
        time.sleep(0.05)
        text = f"[Transformers:{self.entry.alias}] Response to: {req.prompt[:60]}... — generated with 4-bit QLoRA on {self.entry.params} model."
        latency = (time.time() - t0) * 1000 + 120
        return GenResponse(text=text[:req.max_tokens*4], tokens=req.max_tokens//2, latency_ms=latency, vram_mb=self.entry.vram_4bit*1024, backend="transformers", model=self.entry.alias)

class LlamaCppAdapter(PowerInferAdapter):
    def generate(self, req: GenRequest) -> GenResponse:
        r = super().generate(req)
        r.backend = "llama.cpp"
        r.vram_mb = 6200
        return r

class VLLMAdapter(BaseAdapter):
    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.time()
        time.sleep(0.02)
        text = f"[vLLM PagedAttention] High-throughput response for {self.entry.alias} — continuous batching, 2.8x throughput."
        return GenResponse(text=text[:req.max_tokens*4], tokens=req.max_tokens, latency_ms=45, vram_mb=7000, backend="vllm", model=self.entry.alias)

class EmbeddingAdapter(BaseAdapter):
    def embed(self, texts: List[str]) -> List[List[float]]:
        import random, math
        # Simulate embeddings
        return [[random.uniform(-1,1) for _ in range(384)] for _ in texts]
    def generate(self, req: GenRequest) -> GenResponse:
        # For embed models, generate not applicable, but we simulate
        return GenResponse(text="[Embed] Use /v1/embeddings", tokens=0, latency_ms=10, backend="embedding", model=self.entry.alias)

def get_adapter(entry, backend_hint="auto", hw_vram=6) -> BaseAdapter:
    from .registry import ModelRegistry
    reg = ModelRegistry()
    backend = backend_hint if backend_hint != "auto" else reg.auto_backend(entry, hw_vram)
    task = entry.task
    if task == "embed":
        return EmbeddingAdapter(entry)
    if backend == "powerinfer":
        return PowerInferAdapter(entry)
    elif backend == "llama.cpp":
        return LlamaCppAdapter(entry)
    elif backend == "vllm":
        return VLLMAdapter(entry)
    else:
        return TransformersAdapter(entry)
