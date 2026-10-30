# PowerInfer 2.0 — Design Principles & Architecture

## Key Design Principles (from spec)

1. **Correctness over speed:** Never sacrifice output quality. If sparsity degrades perplexity by >0.5%, fall back to dense.
2. **Graceful degradation:** Every component must have a dense fallback. The engine must never crash due to sparsity assumptions.
3. **Zero-copy where possible:** Minimize CPU↔GPU transfers. Hot weights stay on GPU permanently. Cold weights transfer only when needed.
4. **Predictor overhead budget:** The predictor must cost <5% of total inference time (<0.05ms per layer per token). If it costs more, the sparsity savings are negated.
5. **Consumer hardware first:** Optimize for 8–16 GB VRAM, 16–32 GB RAM, PCIe 4.0 x16 bandwidth. Do not assume datacenter interconnects.
6. **Backward compatible:** Sparse GGUF files must be loadable by standard llama.cpp (ignoring sparse metadata) for maximum ecosystem compatibility.

## Architecture Subsystems

### Subsystem A — Offline Neuron Profiler (Python + PyTorch)
- File: `profile.py`
- Accepts any HuggingFace Transformers model (LLaMA, Mistral, Qwen, Falcon, Phi)
- Calibration on 10k+ tokens (C4, RedPajama, or user-provided text)
- Per-FFN activation frequency recording
- Hot/Warm/Cold classification (default 15%/25%/60%)
- Learned predictor for SwiGLU/GELU: 2-layer MLP (hidden→512→ffn) trained 3 epochs BCE, threshold 0.5
- Output: `.predictor` binary (header + per-layer hot lists + MLP weights float16 + global stats)

### Subsystem B — Sparse Weight Layout & Quantization (C++/CUDA + Python)
- Files: `convert_sparse.py`, `convert_sparse`, `include/powerinfer/sparse_gguf.h`
- Extends GGUF with `SPARSE_GGUF`: splits W_gate, W_up, W_down into W_hot (contiguous GPU) and W_cold (CSR CPU pinned)
- Hot quant: Q8_0 or Q6_K (higher precision), Cold quant: Q4_K_M or Q3_K_M (aggressive)
- Packs all W_hot into single contiguous GPU buffer for single `cudaMemcpy`
- CSR for cold enables efficient random access when cold neurons fire
- Backward compatible: dense tensors (attention) remain, sparse metadata ignored by llama.cpp

### Subsystem C — Runtime Predictor (CUDA C++)
- Files: `include/powerinfer/predictor.h`, `src/predictor.cpp`
- ReLU/ReGLU: trivial static mask (hot always, cold never) — zero overhead
- SwiGLU/GELU: loads per-layer MLP from `.predictor`, runs `Linear→ReLU→Linear→Sigmoid` before each FFN
- Fused into preceding layer's output kernel, uses shared memory for 512-dim intermediate, <0.05ms per layer
- Low confidence (0.3–0.7 for many neurons) → graceful fallback to dense
- Output: packed uint32 mask (1 bit per neuron) in GPU shared/L2

### Subsystem D — Sparse FFN CUDA Kernels (Critical Path)
- File: `src/sparse_ffn.cu` (+ `src/sparse_ffn_cpu.cpp` fallback)
- **Kernel 1 `sparse_ffn_hot_only`** (fast path 85% tokens): fuses three matmuls + SiLU, uses WMMA (sm_80+), tcgen05 (sm_120 Blackwell), FP8 E4M3, target occupancy ≥75%
- **Kernel 2 `sparse_ffn_hybrid`** (slow path <15% tokens): async prefetch cold weights on dedicated stream, overlaps with hot compute, LRU cache 512MB, aborts to dense if >30% cold active
- **Kernel 3 `sparse_attention`** (Phase 2): sparsity at head granularity, skips KV-cache for pruned heads
- CPU fallbacks ensure correctness without CUDA

### Subsystem E — Memory Manager & Orchestrator (C++)
- Files: `include/powerinfer/memory_manager.h`, `src/memory_manager.cpp`
- Model load: allocates GPU for attention+W_hot+KV(4096)+LRU(512MB)+predictor, pinned CPU for W_cold+embedding, reports breakdown
- Runtime: CUDA graphs for hot path, async copies on cold stream, speculative prefetch for next token via n-gram, OOM handling (demote warm→cold, swap to SSD, reduce context, actionable advice)

## Memory Layout Example (8GB VRAM)

```
[Memory Layout]
GPU VRAM (8192 MB):
  Attention weights:    1842 MB
  Hot FFN weights:      2104 MB
  KV cache (4096 tok):  1024 MB
  Cold LRU cache:        512 MB
  Predictor MLPs:         48 MB
  Activations/scratch:   662 MB
  ─────────────────────────────
  Total:                6192 MB / 8192 MB (75.6%)

CPU RAM (pinned):
  Cold FFN weights:     3847 MB
  Embedding:             512 MB
  ─────────────────────────────
  Total:                4359 MB
```

## Model Compatibility

- **Fully Supported (ReLU ~90% sparsity)**: ReluLLaMA 7B/13B/33B/65B, LLaMA with ReLU patch, Falcon
- **Via Learned Predictor (moderate)**: LLaMA 2/3/3.1, Mistral/Mixtral, Qwen/Qwen2, Phi-3/4, Gemma/Gemma2
- **MoE (special)**: Mixtral 8x7B/8x22B, Qwen2-MoE, DeepSeek-V2/V3 — sparsity *within* each expert

## Build System

```cmake
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120" \
  -DSPARSE_FFN=ON \
  -DPREDICTOR_MLP=ON \
  -DFP8_SUPPORT=ON \
  -DCOLD_CACHE_SIZE_MB=512
```

- Requires CMake 3.28+, CUDA Toolkit 12.8+ (Blackwell sm_120), GCC 13+/MSVC 2022, Python 3.10+, PyTorch 2.3+
- NVIDIA driver ≥570.0, CUDA runtime 12.8+, no Python at runtime

## CLI

```bash
python profile.py --model meta-llama/Llama-3-8B --dataset c4-validation --tokens 50000 --hot-ratio 0.15 --warm-ratio 0.25 --output llama3-8b.predictor
./convert_sparse --input llama-3-8b-q4_k_m.gguf --predictor llama3-8b.predictor --output llama-3-8b-sparse.gguf --hot-quant q8_0 --cold-quant q3_k_m
./powerinfer -m models/llama-3-8b-sparse.gguf -p "Explain quantum computing:" -n 512 -c 4096 --predictor llama3-8b.predictor --hot-layers all --cold-cache 512 --sparsity-threshold 0.15 --fallback-dense --verbose
```

## Testing

- `tests/test_powerinfer2.py`: 18 tests covering profiler, converter, predictor, kernels, memory, CLI, edge cases
- `scripts/validate_sparse_correctness.py`: perplexity <0.5% and token match >95%
- `benchmark.py`: validates performance targets on RTX 5060/5080/5090

## Phased Delivery (spec)

- Phase 1 (1-6w): Foundation (profiler, converter, hot-only kernel, ReluLLaMA 7B, 2x speedup)
- Phase 2 (7-12w): Full sparsity (learned predictor, hybrid kernel, memory manager, LLaMA3/Qwen2.5, 3x)
- Phase 3 (13-18w): Blackwell FP8, CUDA graphs, MoE, 30B on 5060 at ≥8 tok/s
- Phase 4 (19-24w): Multi-GPU, server mode, dynamic adaptation

See `PowerInfer` original paper: https://arxiv.org/abs/2406.06282 (PowerInfer-2) and https://arxiv.org/abs/2406.05955 (TurboSparse).
