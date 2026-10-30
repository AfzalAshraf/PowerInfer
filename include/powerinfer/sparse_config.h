#pragma once
// PowerInfer 2.0 — Sparse Configuration
// Central configuration for sparsity thresholds and hardware targets

#include <cstdint>
#include <string>

#define POWERINFER_VERSION_MAJOR 2
#define POWERINFER_VERSION_MINOR 0
#define POWERINFER_VERSION_PATCH 0

#define POWERINFER_PREDICTOR_MAGIC "PWR2PRED"
#define POWERINFER_PREDICTOR_VERSION 1

// Supported GPU architectures (spec requirement)
#define POWERINFER_CUDA_ARCH_BLACKWELL 120
#define POWERINFER_CUDA_ARCH_HOPPER    90
#define POWERINFER_CUDA_ARCH_ADA       89
#define POWERINFER_CUDA_ARCH_AMPERE_86 86
#define POWERINFER_CUDA_ARCH_AMPERE_80 80

// Performance constraint: predictor must run <0.05ms per layer per token
#define POWERINFER_PREDICTOR_BUDGET_MS 0.05f

// Default sparsity ratios
#define POWERINFER_DEFAULT_HOT_RATIO 0.15f
#define POWERINFER_DEFAULT_WARM_RATIO 0.25f
#define POWERINFER_DEFAULT_COLD_CACHE_MB 512
#define POWERINFER_DEFAULT_SPARSE_THRESHOLD 0.15f

// Hot weight packing optimization: single cudaMemcpy for all layers
#define POWERINFER_HOT_PACKED_ENABLED 1

// Fallback thresholds
#define POWERINFER_FALLBACK_COLD_RATIO 0.30f  // if >30% cold neurons active, abort sparsity
#define POWERINFER_FALLBACK_CONFIDENCE_LOW 0.3f
#define POWERINFER_FALLBACK_CONFIDENCE_HIGH 0.7f

namespace powerinfer {

enum class ActivationType {
    RELU,
    REGLU,
    GEGLU_RELU,  // ReLU-family (high sparsity ~90%)
    SILU,        // SwiGLU
    GELU,        // SwiGLU/GELU family (moderate sparsity)
    SWIGLU,
    UNKNOWN
};

inline ActivationType activation_from_string(const std::string& s) {
    if (s == "relu" || s == "reglu" || s == "relu2") return ActivationType::RELU;
    if (s == "geglu_relu") return ActivationType::GEGLU_RELU;
    if (s == "gelu") return ActivationType::GELU;
    if (s == "swiglu" || s == "silu" || s == "swish") return ActivationType::SWIGLU;
    return ActivationType::UNKNOWN;
}

inline bool is_relu_family(ActivationType t) {
    return t == ActivationType::RELU || t == ActivationType::REGLU || t == ActivationType::GEGLU_RELU;
}

struct SparseConfig {
    float hot_ratio = POWERINFER_DEFAULT_HOT_RATIO;
    float warm_ratio = POWERINFER_DEFAULT_WARM_RATIO;
    float sparsity_threshold = POWERINFER_DEFAULT_SPARSE_THRESHOLD;
    int cold_cache_size_mb = POWERINFER_DEFAULT_COLD_CACHE_MB;
    bool predictor_mlp_enabled = true;
    bool fp8_enabled = false;
    bool fallback_dense = true;
    bool verbose = false;
    std::string predictor_path;
    std::string hot_quant = "q8_0";
    std::string cold_quant = "q3_k_m";
    std::string hot_layers = "all"; // or "0,1,2,5-10"

    // Runtime
    int n_layers = 32;
    int hidden_dim = 4096;
    int ffn_dim = 11008;
};

struct MemoryBreakdown {
    size_t attn_weights_mb = 0;
    size_t hot_ffn_mb = 0;
    size_t kv_cache_mb = 0;
    size_t cold_lru_mb = 0;
    size_t predictor_mlps_mb = 0;
    size_t activations_mb = 0;
    size_t total_gpu_mb = 0;
    size_t gpu_capacity_mb = 8192;
    size_t cold_ffn_mb = 0;
    size_t embedding_mb = 512;
    size_t total_cpu_mb = 0;
};

} // namespace powerinfer
