#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "include/tensor.hpp"
#include "include/layers.hpp"
#include "kernels/matmul.cpp"
#include "assert.h"

namespace EzTensor{

    FeedForwardLayer::FeedForwardLayer(std::unordered_map<std::string,Tensor> input_weights){
        weights = input_weights;
    };

    Tensor FeedForwardLayer::forward(Tensor& x){
        //self.w1 is weights["down_proj_weight"] 
        //self.w2 is weights["gate_proj_weight"] 
        //self.w3 is weights["up_proj_weight"] 
        MM_MODE mm_mode = MM_MODE::SIMD;
        Tensor w1t = weights["mlp_down_proj_weight"].T();
        Tensor w2t = weights["mlp_down_proj_weight"].T();
        Tensor w3t = weights["mlp_up_proj_weight"].T();
        Tensor o_1 = x.matmul(w1t,mm_mode);
        Tensor o_silu = o_1.silu();
        Tensor o_3 = x.matmul(w3t,mm_mode);
        Tensor o_2 = o_silu.matmul(w2t,mm_mode);
        Tensor output = o_2 * o_3;
        return output;
    };

}