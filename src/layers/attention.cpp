#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "include/tensor.hpp"
#include "include/layers.hpp"
#include "kernels/matmul.cpp"
#include "models/llama.cpp"
#include "assert.h"

namespace EzTensor{

    // dim = params.dim / params.n_heads = head dim
    Tensor AttentionLayer::precompute_freq_cis(int dim, int end, float theta){
        // dim = 64
        std::vector<float> thetas;
        std::vector<float> t;
        for (int i=0 ; i<dim; i+=2){
            thetas.push_back(i);
        }
        std::vector<int> shape = {dim}; 
        Tensor thetas_new(shape, thetas);
        Tensor thetas_div_dim = thetas_new / dim;
        Tensor thetas_div_dim_raise = thetas_div_dim.rpowr(theta);
        for (int i=0; i<end; i++){
            t.push_back(i);
        }
        
    }
    
    AttentionLayer::AttentionLayer(std::unordered_map<std::string,Tensor> input_weights, ModelArgs args){
        weights = input_weights;
        n_kv_heads = args.n_kv_heads;
        n_local_heads = args.n_heads;
        head_dim = args.dim / args.n_heads;
    };

    Tensor AttentionLayer::forward(Tensor& x, int start_pos){
        assert(x.shape.size()==3 && "Invalid shape, expected shape (batch size, seqlen, dim)");
        int batch_size = x.shape[0];
        int seq_len = x.shape[1];
        MM_MODE mm_mode = MM_MODE::SIMD;
        Tensor wqt = weights["self_attn_q_proj_weight"].T();
        Tensor wkt = weights["self_attn_k_proj_weight"].T();
        Tensor wvt = weights["self_attn_v_proj_weight"].T();
        Tensor xq = x.matmul(wqt,mm_mode);
        Tensor xk = x.matmul(wkt,mm_mode);
        Tensor xv = x.matmul(wvt,mm_mode);
        std::vector<int> qkv_shape = {batch_size, seq_len, n_local_heads, head_dim};
        Tensor xq_v = xq.view(qkv_shape);
        Tensor xk_v = xq.view(qkv_shape);
        Tensor xv_v = xq.view(qkv_shape);

    };

}