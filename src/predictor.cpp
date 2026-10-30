#include "powerinfer/predictor.h"
#include "powerinfer/sparse_config.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace powerinfer {

struct RuntimePredictor::Impl {
    // CUDA resources placeholder
    void* stream = nullptr;
    void* d_W1 = nullptr;
    void* d_b1 = nullptr;
    void* d_W2 = nullptr;
    void* d_b2 = nullptr;
};

RuntimePredictor::RuntimePredictor(const SparseConfig& cfg) : config_(cfg), impl_(new Impl()) {
    // Enable CUDA graphs placeholder
}

RuntimePredictor::~RuntimePredictor() = default;

bool RuntimePredictor::load(const std::string& predictor_path) {
    std::ifstream f(predictor_path, std::ios::binary);
    if (!f) {
        std::cerr << "[Predictor] Failed to open " << predictor_path << std::endl;
        // Create dummy predictor for testing
        header_.magic[0]='P'; header_.magic[1]='W'; header_.magic[2]='R'; header_.magic[3]='2';
        header_.magic[4]='P'; header_.magic[5]='R'; header_.magic[6]='E'; header_.magic[7]='D';
        header_.version = 1;
        header_.model_name = config_.predictor_path.empty() ? "dummy" : config_.predictor_path;
        header_.arch = "llama";
        header_.n_layers = config_.n_layers;
        header_.hidden_dim = config_.hidden_dim;
        header_.ffn_dim = config_.ffn_dim;
        layers_.resize(header_.n_layers);
        for (int i=0;i<header_.n_layers;i++) {
            int num_hot = std::max(1, int(header_.ffn_dim * config_.hot_ratio));
            layers_[i].hot_indices.resize(num_hot);
            for (int j=0;j<num_hot;j++) layers_[i].hot_indices[j]=j;
            layers_[i].warm_indices.resize(int(header_.ffn_dim * config_.warm_ratio));
            layers_[i].has_mlp = !is_relu_family(activation_from_string("silu")); // default SwiGLU -> has predictor
            if (layers_[i].has_mlp) {
                layers_[i].mlp.hidden_dim = header_.hidden_dim;
                layers_[i].mlp.ffn_dim = header_.ffn_dim;
                layers_[i].mlp.W1.assign(512*header_.hidden_dim, 0);
                layers_[i].mlp.b1.assign(512, 0);
                layers_[i].mlp.W2.assign(header_.ffn_dim*512, 0);
                layers_[i].mlp.b2.assign(header_.ffn_dim, 0);
            }
        }
        is_relu_model_ = is_relu_family(activation_from_string("relu")) ? true : false; // synthetic
        // For spec: ReLU models have trivial predictor, SwiGLU have learned
        // We'll default to SwiGLU learned predictor for demonstration
        is_relu_model_ = false;
        std::cerr << "[Predictor] Created synthetic predictor for " << header_.n_layers << " layers\n";
        return false;
    }

    // Read header
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "PWR2PRED", 8)!=0) {
        std::cerr << "[Predictor] Invalid magic\n";
        return false;
    }
    uint32_t version; f.read((char*)&version, 4);
    uint32_t mlen; f.read((char*)&mlen,4);
    std::string model_name(mlen, '\0'); f.read(&model_name[0], mlen);
    uint32_t alen; f.read((char*)&alen,4);
    std::string arch(alen, '\0'); f.read(&arch[0], alen);
    int32_t n_layers, hidden_dim, ffn_dim;
    f.read((char*)&n_layers,4); f.read((char*)&hidden_dim,4); f.read((char*)&ffn_dim,4);

    header_.version = version;
    header_.model_name = model_name;
    header_.arch = arch;
    header_.n_layers = n_layers;
    header_.hidden_dim = hidden_dim;
    header_.ffn_dim = ffn_dim;
    layers_.resize(n_layers);

    for (int i=0;i<n_layers;i++) {
        int32_t num_hot; f.read((char*)&num_hot,4);
        layers_[i].hot_indices.resize(num_hot);
        if (num_hot>0) f.read((char*)layers_[i].hot_indices.data(), num_hot*4);
        int32_t num_warm; f.read((char*)&num_warm,4);
        layers_[i].warm_indices.resize(num_warm);
        if (num_warm>0) f.read((char*)layers_[i].warm_indices.data(), num_warm*4);
        uint8_t has_pred; f.read((char*)&has_pred,1);
        layers_[i].has_mlp = has_pred != 0;
        if (has_pred) {
            int32_t h1_out, h1_in; f.read((char*)&h1_out,4); f.read((char*)&h1_in,4);
            layers_[i].mlp.W1.resize(h1_out*h1_in);
            f.read((char*)layers_[i].mlp.W1.data(), h1_out*h1_in*2);
            layers_[i].mlp.b1.resize(h1_out);
            f.read((char*)layers_[i].mlp.b1.data(), h1_out*2);
            int32_t h2_out, h2_in; f.read((char*)&h2_out,4); f.read((char*)&h2_in,4);
            layers_[i].mlp.W2.resize(h2_out*h2_in);
            f.read((char*)layers_[i].mlp.W2.data(), h2_out*h2_in*2);
            layers_[i].mlp.b2.resize(h2_out);
            f.read((char*)layers_[i].mlp.b2.data(), h2_out*2);
            layers_[i].mlp.hidden_dim = h1_in;
            layers_[i].mlp.ffn_dim = h2_out;
            layers_[i].mlp.intermediate_dim = h1_out;
        }
    }
    float sparsity; f.read((char*)&sparsity,4);
    int64_t tokens, timestamp; f.read((char*)&tokens,8); f.read((char*)&timestamp,8);
    float hot_ratio, warm_ratio;
    if (f.read((char*)&hot_ratio,4)) f.read((char*)&warm_ratio,4);

    // Detect ReLU family from arch/model name
    is_relu_model_ = (arch=="bamboo" || arch.find("relu")!=std::string::npos) ? true : is_relu_family(activation_from_string(arch));
    // More accurate: check model_name contains relu
    if (model_name.find("Relu")!=std::string::npos || model_name.find("relu")!=std::string::npos || model_name.find("ProSparse")!=std::string::npos) {
        is_relu_model_ = true;
        // For ReLU, disable MLP even if file says has_mlp? But file should have has_mlp false for ReLU
        // Keep as is
    }

    avg_active_ratio_ = 1.0f - sparsity;
    std::cout << "[Predictor] Loaded " << predictor_path << " arch=" << arch << " layers=" << n_layers << " hidden=" << hidden_dim << " ffn=" << ffn_dim << " relu=" << is_relu_model_ << " sparsity=" << sparsity << "\n";
    return true;
}

RuntimePredictor::PredictionResult RuntimePredictor::predict(int layer_id, const float* hidden_state, int hidden_dim) {
    if (is_relu_model_) {
        return static_predict(layer_id);
    }
    // Use CPU fallback if CUDA not available or for testing
    return predict_cpu(layer_id, hidden_state, hidden_dim);
}

RuntimePredictor::PredictionResult RuntimePredictor::static_predict(int layer_id) {
    return static_predict_cpu(layer_id);
}

RuntimePredictor::PredictionResult RuntimePredictor::static_predict_cpu(int layer_id) {
    PredictionResult res;
    if (layer_id <0 || layer_id >= (int)layers_.size()) return res;
    const auto& lp = layers_[layer_id];
    int ffn_dim = header_.ffn_dim;
    int num_u32 = (ffn_dim+31)/32;
    res.mask.assign(num_u32, 0);
    // Hot neurons always active (1)
    for (int idx : lp.hot_indices) {
        if (idx>=0 && idx<ffn_dim) res.mask[idx/32] |= (1u << (idx%32));
    }
    res.num_active = lp.hot_indices.size();
    res.active_indices = lp.hot_indices; // vector<int32> to int
    std::vector<int> act;
    act.reserve(lp.hot_indices.size());
    for (auto v: lp.hot_indices) act.push_back((int)v);
    res.active_indices = act;
    res.is_low_confidence = false;
    res.confidence = 1.0f;
    res.should_fallback_dense = false;
    total_predictions_++;
    return res;
}

RuntimePredictor::PredictionResult RuntimePredictor::predict_cpu(int layer_id, const float* hidden_state, int hidden_dim) {
    PredictionResult res;
    if (layer_id <0 || layer_id >= (int)layers_.size()) return res;
    const auto& lp = layers_[layer_id];
    if (!lp.has_mlp) {
        return static_predict_cpu(layer_id);
    }
    // Learned predictor: Linear(hidden->512) -> ReLU -> Linear(512->ffn) -> Sigmoid
    // Use float16 weights stored as uint16_t; convert to float for CPU
    auto half_to_float = [](uint16_t h)->float {
        // Simple half conversion via float16 bits (use std::bit_cast if available, manual)
        // For CPU fallback we approximate via conversion: use union
        // This is simplified; for testing we just treat quantized weight as random small value if conversion fails
        // We'll implement proper half decode
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) {
            // subnormal
            if (mant == 0) f = sign << 31;
            else {
                // normalize
                exp = 1;
                while ((mant & 0x400) == 0) { mant <<=1; exp--; }
                mant &= 0x3FF;
                f = (sign<<31) | ((exp + 112)<<23) | (mant<<13);
            }
        } else if (exp == 31) {
            f = (sign<<31) | 0x7F800000 | (mant<<13);
        } else {
            f = (sign<<31) | ((exp+112)<<23) | (mant<<13);
        }
        float out;
        std::memcpy(&out, &f, 4);
        return out;
    };
    int ffn_dim = header_.ffn_dim;
    std::vector<float> h1(512, 0.0f);
    // W1: [512, hidden_dim] row-major
    for (int i=0;i<512;i++) {
        float sum = 0.0f;
        // Add bias b1
        float b = half_to_float(lp.mlp.b1[i]);
        sum += b;
        for (int j=0;j<hidden_dim;j++) {
            uint16_t w = lp.mlp.W1[i*hidden_dim + j];
            sum += half_to_float(w) * hidden_state[j];
        }
        // ReLU
        h1[i] = sum > 0 ? sum : 0;
    }
    std::vector<float> logits(ffn_dim, 0.0f);
    std::vector<float> probs(ffn_dim, 0.0f);
    for (int i=0;i<ffn_dim;i++) {
        float sum = half_to_float(lp.mlp.b2[i]);
        for (int j=0;j<512;j++) {
            uint16_t w = lp.mlp.W2[i*512 + j];
            sum += half_to_float(w) * h1[j];
        }
        logits[i] = sum;
        // Sigmoid
        float p = 1.0f / (1.0f + std::exp(-sum));
        probs[i] = p;
    }
    // Threshold at 0.5
    int num_u32 = (ffn_dim+31)/32;
    res.mask.assign(num_u32, 0);
    int active = 0;
    int low_conf = 0;
    for (int i=0;i<ffn_dim;i++) {
        if (probs[i] > 0.5f) {
            res.mask[i/32] |= (1u << (i%32));
            res.active_indices.push_back(i);
            active++;
        }
        if (probs[i] >= 0.3f && probs[i] <= 0.7f) low_conf++;
    }
    res.num_active = active;
    res.confidence = 1.0f - (float)low_conf / std::max(1, ffn_dim);
    res.is_low_confidence = (low_conf > ffn_dim * 0.2f); // >20% uncertain
    // Graceful degradation: if low confidence for many neurons, fallback to dense
    if (res.is_low_confidence && res.confidence < 0.6f) {
        res.should_fallback_dense = true;
    }
    // Also if active >30% of total, fallback dense per spec
    if (active > ffn_dim * 0.3f) {
        res.should_fallback_dense = true;
    }
    total_predictions_++;
    // Update avg
    avg_active_ratio_ = (avg_active_ratio_ * (total_predictions_-1) + (float)active / ffn_dim) / total_predictions_;
    return res;
}

std::vector<RuntimePredictor::PredictionResult> RuntimePredictor::predict_batch(int layer_id, const float* hidden_states, int batch, int hidden_dim) {
    std::vector<PredictionResult> out;
    out.reserve(batch);
    for (int b=0;b<batch;b++) {
        out.push_back(predict(layer_id, hidden_states + b*hidden_dim, hidden_dim));
    }
    return out;
}

// CUDA fused predictor — stubs that would be compiled with nvcc
// For non-CUDA builds we provide CPU-side no-ops with timing check

namespace cuda {

void launch_fused_predictor(const FusedPredictorArgs& args, void* stream) {
    // Dispatch based on compute capability detected at runtime
    // For host testing, just call dummy
    // Real implementation would:
    // - Use shared memory for 512 intermediate
    // - Fuse into preceding layer output kernel to avoid extra launch
    // - Use cooperative groups
    // Here we simulate <0.05ms budget via chrono
    auto start = std::chrono::high_resolution_clock::now();
    // Simulate work: dummy linear ops
    volatile float dummy = 0;
    for (int i=0;i<512;i++) dummy += 0.001f;
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end-start).count();
    if (ms > POWERINFER_PREDICTOR_BUDGET_MS) {
        // In real code would log that predictor exceeded budget and consider fallback
    }
    // Dispatch to arch specific
#if defined(__CUDACC__)
    // Would call appropriate sm version
#else
    (void)args; (void)stream;
#endif
}

void launch_fused_predictor_sm80(const FusedPredictorArgs& args, void* stream) {
    // Ampere WMMA m16n16k16 implementation
    // Use nvcuda::wmma fragments for Tensor Cores
    // Pseudocode:
    // wmma::fragment<matrix_a, 16,16,16, half, row_major> aFrag;
    // wmma::fragment<matrix_b, 16,16,16, half, col_major> bFrag;
    // wmma::fragment<accumulator, 16,16,16, float> cFrag;
    // wmma::load_matrix_sync(aFrag, hidden_state, 16);
    // wmma::load_matrix_sync(bFrag, W1, 16);
    // wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    // ... ReLU ...
    // Second GEMM similarly
    (void)args; (void)stream;
}

void launch_fused_predictor_sm90(const FusedPredictorArgs& args, void* stream) {
    // Hopper WGMMA (warpgroup matrix multiply accumulate) with TMA
    // Uses Hopper's wgmma.mma_async and TMA async copy
    (void)args; (void)stream;
}

void launch_fused_predictor_sm120(const FusedPredictorArgs& args, void* stream) {
    // Blackwell tcgen05 + FP8 E4M3
    // Uses new tcgen05.mma instructions and FP8 Tensor Cores
    // If model quality allows, W_hot stored as FP8 (E4M3) for 2x throughput
    // Descriptor-based TMA for async shared memory
    // Pseudocode:
    // tcgen05::copy_async(shared_mem, W1_fp8, ...);
    // tcgen05::mma_fp8_e4m3(...);
    (void)args; (void)stream;
}

bool check_low_confidence(const float* sigmoid_outputs, int ffn_dim, float* confidence_out) {
    int low = 0;
    for (int i=0;i<ffn_dim;i++) {
        if (sigmoid_outputs[i] >= POWERINFER_FALLBACK_CONFIDENCE_LOW && sigmoid_outputs[i] <= POWERINFER_FALLBACK_CONFIDENCE_HIGH) low++;
    }
    float conf = 1.0f - (float)low / std::max(1, ffn_dim);
    if (confidence_out) *confidence_out = conf;
    return low > ffn_dim * 0.2f;
}

} // namespace cuda

} // namespace powerinfer
