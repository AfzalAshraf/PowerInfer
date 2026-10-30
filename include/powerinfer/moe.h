#pragma once
// PowerInfer 2.0 — MoE Model Support (Section 3.3)
// For MoE: apply sparsity WITHIN each expert, not across experts (expert routing already provides inter-expert sparsity)

#include "sparse_config.h"
#include "sparse_ffn.h"
#include <vector>
#include <string>

namespace powerinfer {

// MoE models: Mixtral 8x7B, 8x22B, Qwen2-MoE, Qwen3-A3B, DeepSeek-V2/V3
// Strategy: each expert's FFN is sparsely executed via same predictor framework

struct MoEExpert {
    int expert_id = -1;
    // Hot/cold weights per expert (same layout as dense FFN but per expert)
    void* W_gate_hot = nullptr;
    void* W_up_hot = nullptr;
    void* W_down_hot = nullptr;
    void* W_gate_cold = nullptr;
    void* W_up_cold = nullptr;
    void* W_down_cold = nullptr;
    int num_hot = 0;
    int num_cold = 0;
};

struct MoELayer {
    int layer_id = -1;
    int num_experts = 8;
    int top_k = 2; // Mixtral routes to 2 experts per token
    std::vector<MoEExpert> experts;
    // Router (gate) weights
    void* router_weights = nullptr; // [hidden, num_experts]
};

class SparseMoE {
public:
    explicit SparseMoE(const SparseConfig& cfg) : config_(cfg) {}
    
    // Initialize MoE layer with per-expert hot/cold splits
    bool init_layer(int layer_id, int num_experts, int top_k);

    // Forward: route to top_k experts, then apply sparse FFN within each expert
    // Returns sparse output aggregated over experts
    bool forward(int layer_id, const void* hidden_state, void* output, int batch,
                 const int* expert_ids, const float* expert_weights);

    // Apply sparsity within expert: reuse SparseFFN logic per expert
    // e.g., Mixtral 8x7B with 4B active: each expert's FFN is sparsely computed
    // TurboSparse-Mixtral-47B example: 47B total, 4B active via expert sparsity + neuron sparsity

private:
    SparseConfig config_;
    std::vector<MoELayer> layers_;
};

// MoE-aware predictor: predictor now predicts for each expert's neurons separately
// Input: hidden state + expert id -> mask per expert

} // namespace powerinfer
