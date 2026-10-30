#pragma once
// PowerInfer 2.0 — Runtime Predictor (Subsystem C)
// Predicts which neurons will activate for each token before FFN runs.

#include "sparse_config.h"
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace powerinfer {

// Predictor MLP architecture per layer (spec):
//   Linear(hidden_dim -> 512) -> ReLU -> Linear(512 -> ffn_dim) -> Sigmoid
//   Threshold at 0.5 to produce binary mask
//   Must run <0.05ms per layer per token, fused into preceding layer's output kernel
//   Uses shared memory for intermediate 512-dim representation
// For ReLU/ReGLU: trivial predictor (hot always active, cold never) — no runtime cost
// For SwiGLU/GELU: learned predictor loaded from .predictor file

struct PredictorWeights {
    // W1: [512, hidden_dim] float16
    std::vector<uint16_t> W1; // float16 storage, 512*hidden
    std::vector<uint16_t> b1; // 512
    std::vector<uint16_t> W2; // [ffn_dim, 512]
    std::vector<uint16_t> b2; // ffn_dim
    int hidden_dim = 4096;
    int ffn_dim = 11008;
    int intermediate_dim = 512;

    size_t bytes() const {
        return (W1.size()+b1.size()+W2.size()+b2.size())*sizeof(uint16_t);
    }
};

struct PredictorHeader {
    char magic[8] = {'P','W','R','2','P','R','E','D'};
    uint32_t version = 1;
    std::string model_name;
    std::string arch;
    int32_t n_layers = 32;
    int32_t hidden_dim = 4096;
    int32_t ffn_dim = 11008;
};

struct LayerPredictor {
    std::vector<int32_t> hot_indices;  // sorted int32 array, size num_hot
    std::vector<int32_t> warm_indices;
    bool has_mlp = false; // true for SwiGLU/GELU
    PredictorWeights mlp; // valid if has_mlp
    float sparsity_ratio = 0.85f;
};

class RuntimePredictor {
public:
    explicit RuntimePredictor(const SparseConfig& cfg);
    ~RuntimePredictor();

    // Load per-layer predictor MLP from .predictor file
    // Returns true on success, false falls back to static mask
    bool load(const std::string& predictor_path);

    // For ReLU/ReGLU models: trivial static mask (hot neurons always active)
    // For SwiGLU/GELU: run predictor MLP on hidden_state
    // Input: hidden_state [hidden_dim] float16/float32 (current token)
    // Output: binary mask packed into uint32 array (1 bit per neuron)
    //   mask size: (ffn_dim+31)/32 uint32
    // Returns predicted mask vector, and confidence flag
    //   If confidence low (sigmoid 0.3-0.7 for many neurons), caller should fallback to dense
    struct PredictionResult {
        std::vector<uint32_t> mask; // packed bits
        std::vector<int> active_indices; // decoded list for cold path
        bool is_low_confidence = false;
        float confidence = 1.0f;
        int num_active = 0;
        bool should_fallback_dense = false;
    };

    PredictionResult predict(int layer_id, const float* hidden_state, int hidden_dim);

    // Batched version for n tokens
    std::vector<PredictionResult> predict_batch(int layer_id, const float* hidden_states, int batch, int hidden_dim);

    // Static mask for ReLU (no MLP): hot = 1, cold = 0
    PredictionResult static_predict(int layer_id);

    // Metrics
    float avg_active_ratio() const { return avg_active_ratio_; }
    int total_predictions() const { return total_predictions_; }

    const PredictorHeader& header() const { return header_; }
    const LayerPredictor& layer_predictor(int lid) const { return layers_[lid]; }
    bool is_relu_model() const { return is_relu_model_; }
    int n_layers() const { return (int)layers_.size(); }

private:
    SparseConfig config_;
    PredictorHeader header_;
    std::vector<LayerPredictor> layers_;
    bool is_relu_model_ = false;
    float avg_active_ratio_ = 0.15f;
    int total_predictions_ = 0;

    // GPU resources (if CUDA enabled)
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // CPU fallback kernels
    PredictionResult predict_cpu(int layer_id, const float* hidden_state, int hidden_dim);
    PredictionResult static_predict_cpu(int layer_id);
};

// Fused predictor kernel interface (CUDA)
// Must run <0.05ms per layer per token, fused into preceding layer's output
namespace cuda {

// Fuse predictor into preceding layer's output: takes previous layer output directly from registers/shared memory
// Uses shared memory for intermediate 512-dim representation to avoid global memory
// Called before each FFN layer

struct FusedPredictorArgs {
    const void* hidden_state; // half* hidden_dim
    const void* W1; // half [512, hidden_dim]
    const void* b1; // half [512]
    const void* W2; // half [ffn_dim, 512]
    const void* b2; // half [ffn_dim]
    uint32_t* mask_out; // packed uint32 [ffn_dim/32]
    int hidden_dim;
    int ffn_dim;
    float threshold; // 0.5
};

// Generic (sm_80+ uses WMMA, sm_120 uses tcgen05)
void launch_fused_predictor(const FusedPredictorArgs& args, void* stream);

// Specializations
void launch_fused_predictor_sm80(const FusedPredictorArgs& args, void* stream);
void launch_fused_predictor_sm90(const FusedPredictorArgs& args, void* stream);
void launch_fused_predictor_sm120(const FusedPredictorArgs& args, void* stream); // Blackwell tcgen05 + FP8

// Confidence check: if many sigmoids in [0.3, 0.7], flag low confidence
bool check_low_confidence(const float* sigmoid_outputs, int ffn_dim, float* confidence_out);

} // namespace cuda

} // namespace powerinfer
