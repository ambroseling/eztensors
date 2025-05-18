
#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "tensor.cpp"

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
            AttentionLayer(std::unordered_map<std::string,Tensor> input_weights, ModelArgs args);
            ~AttentionLayer();
            Tensor forward(Tensor& x, int start_pos);
            constexpr Tensor precompute_freq_cis(int dim, int end, float theta);
    };


    class RMSNormLayer : public Layer<RMSNormLayer> {
        public: 
            float eps;
            Tensor _norm(Tensor& x);
            RMSNormLayer(std::unordered_map<std::string,Tensor> input_weights);
            ~RMSNormLayer();
            Tensor forward(Tensor& x);


    };

    class FeedForwardLayer : public Layer<FeedForwardLayer> {
        public: 
            int dim;
            int hidden_dim;
            int mulitple_of;
            FeedForwardLayer(std::unordered_map<std::string,Tensor> input_weights);
            ~FeedForwardLayer();
            Tensor forward(Tensor& x);
    };

}