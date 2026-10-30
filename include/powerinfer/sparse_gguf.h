#pragma once
// PowerInfer 2.0 — SPARSE_GGUF Extension
// Extends GGUF format with sparse weight layout

#include "sparse_config.h"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace powerinfer {

// SPARSE_GGUF tensor naming conventions
// Original dense tensor: blk.{bid}.ffn_gate.weight  [ffn_dim, hidden_dim]
// Split into:
//   blk.{bid}.ffn_gate_hot.weight   [num_hot, hidden_dim]  contiguous GPU-resident, row-major matching hot list
//   blk.{bid}.ffn_gate_cold.weight  [num_cold, hidden_dim] CSR compressed, CPU pinned
// Same for ffn_up, ffn_down ([hidden_dim, num_hot] for down)
// W_hot for ALL layers packed into single contiguous GPU buffer for single cudaMemcpy batching
// W_cold remains in pinned host memory for async transfer

struct SparseTensorInfo {
    std::string original_name; // e.g. blk.0.ffn_gate.weight
    std::string hot_name;      // blk.0.ffn_gate_hot.weight
    std::string cold_name;     // blk.0.ffn_gate_cold.weight
    int layer_id = -1;
    std::string ffn_type; // gate/up/down
    int num_hot = 0;
    int num_cold = 0;
    int hidden_dim = 4096;
    int ffn_dim = 11008;
    // Hot stored row-major, contiguous
    bool hot_contiguous = true;
    // Cold stored CSR
    bool cold_csr = true;
    // Quantization
    std::string hot_quant = "q8_0";
    std::string cold_quant = "q3_k_m";
    // CSR auxiliary tensors for cold (if csr)
    std::string cold_csr_indptr_name; // blk.0.ffn_gate_cold.csr_indptr
    std::string cold_csr_indices_name;
    std::string cold_csr_data_name;
};

struct SparseGGUFMetadata {
    bool is_sparse = false;
    std::string predictor_path;
    float sparsity_ratio = 0.85f;
    int n_hot_avg = 0;
    std::string hot_quant;
    std::string cold_quant;
    std::string cold_format = "csr"; // csr
    int cold_cache_mb = 512;
    int hidden_dim = 4096;
    int ffn_dim = 11008;
    int n_layers = 32;
    // Per-layer hot indices
    std::vector<std::vector<int32_t>> hot_indices_per_layer;
    std::unordered_map<std::string, SparseTensorInfo> sparse_tensors;
};

// GGUF keys for sparse extension
namespace gguf_keys {
    constexpr const char* SPARSE = "powerinfer.sparse";
    constexpr const char* SPARSE_PREDICTOR = "powerinfer.sparse.predictor";
    constexpr const char* SPARSE_SPARSITY = "powerinfer.sparse.sparsity";
    constexpr const char* SPARSE_HOT_QUANT = "powerinfer.sparse.hot_quant";
    constexpr const char* SPARSE_COLD_QUANT = "powerinfer.sparse.cold_quant";
    constexpr const char* SPARSE_COLD_FORMAT = "powerinfer.sparse.cold_format";
    constexpr const char* SPARSE_COLD_CACHE_MB = "powerinfer.sparse.cold_cache_mb";
    constexpr const char* SPARSE_HIDDEN_DIM = "powerinfer.sparse.hidden_dim";
    constexpr const char* SPARSE_FFN_DIM = "powerinfer.sparse.ffn_dim";
    constexpr const char* SPARSE_LAYER_HOT_INDICES_PREFIX = "powerinfer.sparse.layer";
    inline std::string layer_hot_key(int layer) {
        return std::string("powerinfer.sparse.layer") + std::to_string(layer) + ".hot_indices";
    }
    inline std::string layer_num_hot_key(int layer) {
        return std::string("powerinfer.sparse.layer") + std::to_string(layer) + ".num_hot";
    }
}

// Helper to detect sparse GGUF file (backward compatible: standard llama.cpp ignores sparse metadata)
bool is_sparse_gguf(const std::string& path);
SparseGGUFMetadata read_sparse_metadata(const std::string& path);

// CSR utilities for cold weights
struct CSRMatrix {
    std::vector<int> indptr;   // size n_rows+1
    std::vector<int> indices;  // column indices of non-zeros
    std::vector<uint16_t> data; // float16 values (or quantized)
    int n_rows = 0;
    int n_cols = 0;
    size_t bytes() const {
        return indptr.size()*sizeof(int) + indices.size()*sizeof(int) + data.size()*sizeof(uint16_t);
    }
    // Random access to cold row: copy row into dense buffer
    void decompress_row(int row, float* out, int hidden_dim) const;
};

CSRMatrix compress_to_csr(const float* dense, int n_rows, int n_cols, float sparsity_threshold = 0.01f);

} // namespace powerinfer
