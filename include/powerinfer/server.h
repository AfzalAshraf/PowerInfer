#pragma once
// PowerInfer 2.0 — Server Mode (Phase 4)
// OpenAI-compatible API with sparse acceleration

#include <string>
#include <vector>

namespace powerinfer {

// Server mode: exposes OpenAI-compatible REST API
// POST /v1/completions, /v1/chat/completions, /v1/embeddings
// Handles continuous batching with sparse FFN per request

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string model_path;
    std::string predictor_path;
    int threads = 8;
    int ctx_size = 4096;
    bool enable_sparse = true;
    // Multi-GPU tensor parallelism across 2x consumer GPUs
    int tensor_parallel = 1; // 1 or 2
};

class SparseServer {
public:
    explicit SparseServer(const ServerConfig& cfg);
    bool start();
    void stop();
    // Request handlers
    std::string handle_completion(const std::string& prompt, int n_predict, float temp);
    std::string handle_chat(const std::string& messages_json);
};

// Dynamic sparsity adaptation: adjust hot/cold ratio at runtime based on load
// e.g., under high load, reduce hot ratio to save VRAM; under low load, increase for quality
void adapt_sparsity_runtime(float gpu_util, float *hot_ratio);

} // namespace powerinfer
