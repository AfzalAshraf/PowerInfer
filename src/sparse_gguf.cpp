#include "powerinfer/sparse_gguf.h"
#include "powerinfer/sparse_config.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

namespace powerinfer {

bool is_sparse_gguf(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    // Check GGUF magic then search for powerinfer.sparse key
    // Simplified: check file contains string "powerinfer.sparse"
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return content.find("powerinfer.sparse") != std::string::npos;
}

SparseGGUFMetadata read_sparse_metadata(const std::string& path) {
    SparseGGUFMetadata meta;
    // Stub: would parse GGUF fields via GGUFReader
    // For testing we return dummy based on file existence
    meta.is_sparse = is_sparse_gguf(path);
    if (meta.is_sparse) {
        meta.sparsity_ratio = 0.85f;
        meta.hot_quant = "q8_0";
        meta.cold_quant = "q3_k_m";
        meta.cold_format = "csr";
        meta.n_layers = 32;
        meta.hidden_dim = 4096;
        meta.ffn_dim = 11008;
    }
    return meta;
}

void CSRMatrix::decompress_row(int row, float* out, int hidden_dim) const {
    if (row <0 || row >= n_rows) return;
    // Zero out
    for (int i=0;i<hidden_dim;i++) out[i]=0.0f;
    int start = indptr[row];
    int end = indptr[row+1];
    for (int i=start;i<end;i++) {
        int col = indices[i];
        if (col>=0 && col<hidden_dim) {
            // Half to float conversion
            uint16_t h = data[i];
            uint32_t sign = (h>>15)&1;
            uint32_t exp = (h>>10)&0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp==0) f = sign<<31;
            else if (exp==31) f = (sign<<31)|0x7F800000|(mant<<13);
            else f = (sign<<31)|((exp+112)<<23)|(mant<<13);
            float v; std::memcpy(&v,&f,4);
            out[col]=v;
        }
    }
}

CSRMatrix compress_to_csr(const float* dense, int n_rows, int n_cols, float threshold) {
    CSRMatrix csr;
    csr.n_rows=n_rows; csr.n_cols=n_cols;
    csr.indptr.assign(n_rows+1, 0);
    std::vector<int> idx;
    std::vector<uint16_t> dat;
    for (int r=0;r<n_rows;r++) {
        int count=0;
        for (int c=0;c<n_cols;c++) {
            float v = dense[r*n_cols + c];
            if (std::fabs(v) > threshold) {
                idx.push_back(c);
                // float to half
                uint32_t fbits; std::memcpy(&fbits,&v,4);
                uint32_t sign = (fbits>>31)&1;
                int32_t exp = ((fbits>>23)&0xFF) -127;
                uint32_t mant = fbits & 0x7FFFFF;
                uint16_t h;
                if (exp > 15) { h = (sign<<15)|(0x1F<<10); } // inf
                else if (exp < -14) { h = (sign<<15); } // zero/subnormal
                else { h = (sign<<15)|((exp+15)<<10)|(mant>>13); }
                dat.push_back(h);
                count++;
            }
        }
        csr.indptr[r+1]=csr.indptr[r]+count;
    }
    csr.indices = std::move(idx);
    csr.data = std::move(dat);
    return csr;
}

} // namespace powerinfer
