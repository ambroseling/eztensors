
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
            void forward(){
                static_cast<Derived*>(this)->forward();
            };
    };

    class AttentionLayer : public Layer<AttentionLayer> {
        public: 
            int num_kv_heads;
            int n_local_heads;
            int n_rep;
            int head_dim;
            void forward();


    };


    class RMSNormLayer : public Layer<RMSNormLayer> {
        public: 
            float eps;
            Tensor _norm(Tensor x);
            void forward();


    };

    class FeedForwardLayer : public Layer<FeedForwardLayer> {
        public: 
            int dim;
            int hidden_dim;
            int mulitple_of;
            void forward();


    };

}