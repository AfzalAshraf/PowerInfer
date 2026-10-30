#pragma once
// PowerInfer 2.0 — Memory Manager & GPU/CPU Orchestrator (Subsystem E)
// Manages split memory layout and dynamic data movement

#include "sparse_config.h"
#include "sparse_gguf.h"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace powerinfer {

// LRU cache for cold weights (GPU)
struct LRUCache {
    struct Entry {
        int layer_id = -1;
        int neuron_id = -1;
        std::vector<uint16_t> data; // float16 weight row
        uint64_t last_access = 0;
        bool dirty = false;
    };
    size_t capacity_bytes = 512 * 1024 * 1024; // default 512 MB
    size_t used_bytes = 0;
    std::unordered_map<uint64_t, Entry> map; // key = (layer<<32 | neuron)
    uint64_t access_counter = 0;
    std::mutex mu;

    bool contains(int layer, int neuron) const;
    const Entry* get(int layer, int neuron);
    void put(int layer, int neuron, const void* data, size_t bytes);
    void evict_lru(size_t need_bytes);
    float hit_rate() const;
    void clear();
};

class MemoryManager {
public:
    explicit MemoryManager(const SparseConfig& cfg);
    ~MemoryManager();

    // At model load time:
    // 1. Read sparse GGUF file
    // 2. Allocate GPU VRAM for attention, W_hot, KV cache, LRU, predictor
    // 3. Allocate pinned CPU memory for W_cold (CSR), embedding, output projection if needed
    // 4. Report memory breakdown
    struct LoadResult {
        bool success = false;
        std::string error;
        MemoryBreakdown breakdown;
    };
    LoadResult load_model(const std::string& gguf_path, const std::string& predictor_path);

    // Runtime orchestration
    // Use CUDA graphs for hot-only fast path (eliminates kernel launch overhead)
    void enable_cuda_graphs(bool enable);
    bool cuda_graphs_enabled() const { return use_cuda_graphs_; }

    // Async prefetch cold weights on dedicated stream
    // Returns true if hit in LRU cache (no transfer needed)
    bool prefetch_cold_weights(int layer_id, const std::vector<int>& neuron_ids, void* stream);

    // Speculative prefetch for next token N+1 while generating N
    // Based on n-gram patterns in predictor activation history
    void speculative_prefetch(int next_layer, const std::vector<int>& predicted_neurons);

    // OOM handling
    // If VRAM exhausted, dynamically demote warm neurons to cold
    // If CPU RAM exhausted, swap cold weights to SSD (memory-mapped file)
    enum class OOMAction {
        NONE,
        DEMOTE_WARM_TO_COLD,
        SWAP_TO_SSD,
        REDUCE_CONTEXT,
        ABORT
    };
    OOMAction handle_oom(size_t requested_bytes, size_t available_bytes);
    std::string oom_advice(size_t requested, size_t available) const;

    // Allocators
    void* allocate_gpu(size_t bytes, const std::string& tag);
    void* allocate_pinned_cpu(size_t bytes, const std::string& tag);
    void free_gpu(void* ptr);
    void free_pinned_cpu(void* ptr);

    // Info
    size_t gpu_total_bytes() const { return gpu_total_; }
    size_t gpu_used_bytes() const { return gpu_used_; }
    size_t gpu_free_bytes() const { return gpu_total_ > gpu_used_ ? gpu_total_ - gpu_used_ : 0; }
    size_t cpu_total_bytes() const { return cpu_total_; }
    size_t cpu_used_bytes() const { return cpu_used_; }

    const MemoryBreakdown& breakdown() const { return breakdown_; }
    void print_breakdown() const;

    LRUCache& lru_cache() { return lru_; }
    const LRUCache& lru_cache() const { return lru_; }

    // For testing without CUDA: simulate pinned memory with regular host allocation
    bool use_pinned_memory() const { return use_pinned_; }

private:
    SparseConfig config_;
    MemoryBreakdown breakdown_;
    LRUCache lru_;
    bool use_cuda_graphs_ = true;
    bool use_pinned_ = true;
    size_t gpu_total_ = 8192ULL * 1024 * 1024;
    size_t gpu_used_ = 0;
    size_t cpu_total_ = 32ULL * 1024 * 1024 * 1024;
    size_t cpu_used_ = 0;
    std::unordered_map<void*, size_t> allocations_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    void update_breakdown();
    void log_breakdown() const;
};

} // namespace powerinfer
