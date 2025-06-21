#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "../include/layers.hpp"
#include "assert.h"

namespace EzTensor{

    AttentionLayer::~AttentionLayer(){
        weights.clear();
    }
    
    Tensor AttentionLayer::precompute_freq_cis(int dim, int end, float theta){
        std::vector<float> thetas;
        std::vector<float> t_d;
        for (int i=0 ; i<dim; i+=2)
            thetas.push_back(i);
        for (int i=0; i<end; i++)
            t_d.push_back(i);     
        std::vector<int> shape = {dim/2}; 
        Tensor thetas_new(shape, thetas);
        Tensor thetas_div_dim = thetas_new / dim;
        Tensor thetas_div_dim_raise = thetas_div_dim.rpowr(theta);
        Tensor inverse_thetas_div_dim_raise = thetas_div_dim_raise.powr(-1.0);
        std::vector<int> t_shape = {end};
        Tensor t(t_shape, t_d);
        Tensor freqs = t.outer(inverse_thetas_div_dim_raise, MM_MODE::SIMD);
        return freqs;
    }

    std::unordered_map<std::string,EzTensor::Tensor> AttentionLayer::apply_rotary_embedding(Tensor& xq, Tensor& xk, Tensor& freqs){
        std::vector<int> shape = {xq.shape[0], xq.shape[1], n_local_heads, head_dim};
        xq = xq.view(shape);
        xk = xk.view(shape);
        Tensor freq_cis_cos_sin = freqs.complex_cos_sin();
        Tensor freq_cis_sin_cos = freqs.complex_sin_cos();
        std::vector<int> broadcast_shape = {1, xq.shape[1], 1, head_dim };
        Tensor freq_cis_cos_sin_b = freq_cis_cos_sin.view(broadcast_shape);
        Tensor freq_cis_sin_cos_b = freq_cis_sin_cos.view(broadcast_shape);
        Tensor out_real_q_cis = xq * freq_cis_cos_sin_b;
        Tensor out_imag_q_cis = xq * freq_cis_sin_cos_b;
        Tensor out_real_k_cis = xk * freq_cis_cos_sin_b;
        Tensor out_imag_k_cis = xk * freq_cis_sin_cos_b;
        std::vector<int> new_shape = {xq.shape[0], xq.shape[1], n_local_heads, head_dim / 2};
        Tensor out_real_q_1 = out_real_q_cis.last_dim_subtract(new_shape);
        Tensor out_imag_q_1 = out_imag_q_cis.last_dim_subtract(new_shape);
        Tensor out_real_k_1 = out_real_k_cis.last_dim_subtract(new_shape);
        Tensor out_imag_k_1 = out_imag_k_cis.last_dim_subtract(new_shape);
        Tensor out_q = out_real_q_1.concat_with(out_imag_q_1, -1);
        Tensor out_k = out_real_k_1.concat_with(out_imag_k_1, -1);
        std::unordered_map<std::string,EzTensor::Tensor> rotary_embeddings;
        rotary_embeddings["out_q"] = out_q;
        rotary_embeddings["out_k"] = out_k;
        return rotary_embeddings;
    }

    AttentionLayer::AttentionLayer(std::unordered_map<std::string,EzTensor::Tensor>& input_weights, ModelArgs args) {
        weights = input_weights;
        n_kv_heads = args.n_kv_heads;
        n_local_heads = args.n_heads;
        head_dim = args.dim / args.n_heads;
        rope_theta = args.rope_theta;
        max_seq_len = args.max_seq_len;
        std::vector<int> shape_k = {args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim};
        std::vector<int> shape_v = {args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim};
        Tensor c_k(shape_k);
        Tensor c_v(shape_v);
        cache_k = &c_k;
        cache_v = &c_v;
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
        Tensor freqs = precompute_freq_cis(head_dim, max_seq_len * 2, rope_theta);
        std::unordered_map<std::string,EzTensor::Tensor>  rot_qk = apply_rotary_embedding(xq,xk,freqs);
        Tensor q_rot = rot_qk["out_q"];
        Tensor k_rot = rot_qk["out_k"];
        return xv_v;
    };

}