
#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "../include/tensor.hpp"
#include "../models/llama.cpp"

namespace EzTensor{

    template <typename Derived>
    class Layer {
        public:
            std::unordered_map<std::string,Tensor> weights;
            Tensor forward(Tensor& x){
                static_cast<Derived*>(this)->forward();
            };
    };

    class AttentionLayer : public Layer<AttentionLayer> {
        public: 
            int n_kv_heads;
            int n_local_heads;
            int n_rep;
            int head_dim;
            int max_seq_len;
            float rope_theta;

            Tensor* cache_k;
            Tensor* cache_v;
            AttentionLayer(std::unordered_map<std::string,Tensor>& input_weights, ModelArgs args);
            ~AttentionLayer();
            Tensor forward(Tensor& x, int start_pos);
            Tensor precompute_freq_cis(int dim, int end, float theta);
            std::unordered_map<std::string,EzTensor::Tensor> apply_rotary_embedding(Tensor& xq, Tensor& xk, Tensor& freqs);
    };


    class RMSNormLayer : public Layer<RMSNormLayer> {
        public: 
            float eps;
            Tensor _norm(Tensor& x);
            RMSNormLayer(std::unordered_map<std::string,Tensor> input_weights, ModelArgs args);
            ~RMSNormLayer();
            Tensor forward(Tensor& x);
    };

    class FeedForwardLayer : public Layer<FeedForwardLayer> {
        public: 
            FeedForwardLayer(std::unordered_map<std::string,Tensor>& input_weights, ModelArgs args);
            ~FeedForwardLayer();
            Tensor forward(Tensor& x);
    };


}