#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "../include/layers.hpp"
#include "assert.h"

namespace EzTensor{

   FeedForwardLayer::~FeedForwardLayer(){
        weights.clear();
    };

    FeedForwardLayer::FeedForwardLayer(std::unordered_map<std::string,Tensor>& input_weights, ModelArgs args){        
        weights = input_weights;
    };

    Tensor FeedForwardLayer::forward(Tensor& x){
        //self.w1 is weights["down_proj_weight"] 
        //self.w2 is weights["gate_proj_weight"] 
        //self.w3 is weights["up_proj_weight"] 
        MM_MODE simd = MM_MODE::SIMD;
        MM_MODE naive = MM_MODE::NAIVE;
        Tensor w1t = weights["mlp_down_proj_weight"].T().contiguous();
        Tensor w2t = weights["mlp_gate_proj_weight"].T().contiguous();
        Tensor w3t = weights["mlp_up_proj_weight"].T().contiguous();
        Tensor o_1 = x.matmul(w2t,simd);
        Tensor o_silu = o_1.silu();
        Tensor o_3 = x.matmul(w3t,naive);
        o_silu.print_tensor(false);
        Tensor o_2 = o_silu.matmul(w1t,naive);
        Tensor output = o_2 * o_3;
        return output;
    };

}