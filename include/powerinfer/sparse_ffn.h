#pragma once
// PowerInfer 2.0 — Sparse FFN CUDA Kernels (Subsystem D — THE CRITICAL PATH)
// Implements sparse_ffn_hot_only, sparse_ffn_hybrid, sparse_attention

#include "sparse_config.h"
#include "predictor.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace powerinfer {

// Forward declarations
class MemoryManager;

// Kernel 1: sparse_ffn_hot_only (Fast Path — 85%+ of tokens)
// Input: hidden_state [batch, hidden_dim]
//        W_hot_gate [num_hot, hidden_dim] (GPU VRAM)
//        W_hot_up [num_hot, hidden_dim]
//        W_hot_down [hidden_dim, num_hot]
//        mask_hot [num_hot] (all 1s)
// Compute:
//   gate = hidden_state * W_hot_gate^T   // [batch, num_hot]
//   up   = hidden_state * W_hot_up^T     // [batch, num_hot]
//   intermediate = SiLU(gate) ⊙ up       // [batch, num_hot]
//   output = intermediate * W_hot_down^T // [batch, hidden_dim]
// Optimization:
//   - Fuse all three matmuls into single kernel launch
//   - Use Tensor Cores (WMMA API) for sm_80+
//   - For sm_120 (Blackwell): tcgen05 Tensor Core instructions
//   - Use FP8 (E4M3) for W_hot on Blackwell if quality allows
//   - Target occupancy ≥75% on all SMs

struct SparseFFNArgs {
    const void* hidden_state; // [batch, hidden_dim] half/bf16
    const void* W_hot_gate;   // [num_hot, hidden_dim]
    const void* W_hot_up;     // [num_hot, hidden_dim]
    const void* W_hot_down;   // [hidden_dim, num_hot] (or transposed)
    void* output;             // [batch, hidden_dim]
    const uint32_t* mask_hot; // packed mask (optional, all 1s for hot-only)
    // For hybrid with cold
    const void* W_cold_gate;  // optional, for hybrid
    const void* W_cold_up;
    const void* W_cold_down;
    const uint32_t* mask_cold; // packed mask for cold
    int batch = 1;
    int hidden_dim = 4096;
    int ffn_dim = 11008;
    int num_hot = 1651; // 15% of 11008 ~1651
    int num_cold_active = 0; // for hybrid: number of cold neurons predicted active
    // FP8 on Blackwell
    bool use_fp8 = false;
    // Stream for async
    void* stream = nullptr; // cudaStream_t
    void* cold_stream = nullptr; // separate stream for cold prefetch
    // LRU cache for cold weights
    void* cold_lru_cache = nullptr;
    size_t cold_lru_bytes = 512 * 1024 * 1024;
};

namespace cuda {

// Core kernels — each has arch-specific implementations

// Hot-only fused kernel
void sparse_ffn_hot_only(const SparseFFNArgs& args);
void sparse_ffn_hot_only_sm80(const SparseFFNArgs& args);   // Ampere WMMA
void sparse_ffn_hot_only_sm89(const SparseFFNArgs& args);   // Ada Lovelace
void sparse_ffn_hot_only_sm90(const SparseFFNArgs& args);   // Hopper WGMMA
void sparse_ffn_hot_only_sm120(const SparseFFNArgs& args);  // Blackwell tcgen05 + FP8

// Hybrid kernel (hot + cold) with async prefetch
// 1. Run hot path
// 2. Identify cold neurons needed from predictor mask
// 3. Async prefetch cold weights from pinned CPU memory via cudaMemcpyAsync on cold_stream
// 4. Compute cold contribution
// 5. Accumulate output += output_cold
// Optimizations: overlap transfer with hot compute (dual streams), LRU cache, fallback if >30%
void sparse_ffn_hybrid(const SparseFFNArgs& args);
void sparse_ffn_hybrid_sm80(const SparseFFNArgs& args);
void sparse_ffn_hybrid_sm90(const SparseFFNArgs& args);
void sparse_ffn_hybrid_sm120(const SparseFFNArgs& args);

// Optional sparse_attention (Phase 2)
// Apply sparsity to attention heads (some heads consistently inactive)
// Skip KV-cache computation for pruned heads
struct SparseAttentionArgs {
    const void* q; // [batch, n_heads, head_dim]
    const void* k;
    const void* v;
    void* output;
    const uint32_t* head_mask; // 1 bit per head
    int batch = 1;
    int n_heads = 32;
    int n_kv_heads = 32;
    int head_dim = 128;
    int seq_len = 4096;
    void* stream = nullptr;
};
void sparse_attention(const SparseAttentionArgs& args);
void sparse_attention_sm80(const SparseAttentionArgs& args);
void sparse_attention_sm120(const SparseAttentionArgs& args);

// Utility: fused SiLU (Swish) + elementwise multiply
// intermediate = silu(gate) * up
// silu(x) = x * sigmoid(x)
void fused_silu_mul(const void* gate, const void* up, void* out, int n, void* stream);

// Tensor Core helpers
// WMMA API wrapper for sm_80+ (m16n16k16)
namespace wmma {
    void gemm_f16_f16(const void* A, const void* B, void* C, int M, int N, int K, void* stream);
    void gemm_fp8_f16(const void* A_fp8, const void* B_f16, void* C, int M, int N, int K, void* stream); // Blackwell
}
// Blackwell tcgen05
namespace tcgen05 {
    void gemm_tcgen05(const void* A, const void* B, void* C, int M, int N, int K, void* stream);
}

} // namespace cuda

// CPU fallback (for testing without CUDA or when VRAM OOM)
namespace cpu {
    void sparse_ffn_hot_only_cpu(const SparseFFNArgs& args);
    void sparse_ffn_hybrid_cpu(const SparseFFNArgs& args);
    void sparse_ffn_dense_cpu(const SparseFFNArgs& args); // fallback when sparsity aborts (>30% cold)
} // namespace cpu

// High-level SparseFFN orchestrator — chooses hot-only vs hybrid vs dense fallback
class SparseFFN {
public:
    explicit SparseFFN(const SparseConfig& cfg, MemoryManager* mem_mgr, RuntimePredictor* predictor);
    ~SparseFFN();

    // Initialize per-layer weights (hot already on GPU, cold in pinned host)
    bool init_layer(int layer_id, const void* w_gate_hot, const void* w_up_hot, const void* w_down_hot,
                    size_t hot_bytes, const void* w_gate_cold, const void* w_up_cold, const void* w_down_cold,
                    size_t cold_bytes);

    // Forward for one token (or batch)
    // hidden_state: [batch, hidden_dim] input
    // output: [batch, hidden_dim] result
    // mask: predictor output (packed bits) — if nullptr uses static hot mask
    // Returns true if sparse path used, false if fell back to dense
    bool forward(int layer_id, const void* hidden_state, void* output, int batch,
                 const uint32_t* mask, float* sparsity_stats = nullptr);

    // Statistics
    struct Stats {
        int hot_tokens = 0;
        int hybrid_tokens = 0;
        int dense_fallback_tokens = 0;
        float avg_active_ratio = 0.15f;
        float avg_speedup = 3.0f;
    };
    Stats stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

private:
    SparseConfig config_;
    MemoryManager* mem_mgr_;
    RuntimePredictor* predictor_;
    Stats stats_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace powerinfer
