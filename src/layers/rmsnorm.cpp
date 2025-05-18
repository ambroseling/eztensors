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

    RMSNormLayer::RMSNormLayer(std::unordered_map<std::string,Tensor> input_weights){
        eps = 0.000001;
        weights = input_weights;
        assert(weights["input_layernorm_weight"].shape == std::vector<int>{2048});
    };

    Tensor RMSNormLayer::_norm(Tensor& x){
        EzTensor::Tensor x_pow = x.powr(2);
        EzTensor::Tensor x_mean = x_pow.mean(-1,true);
        x_mean += eps;
        EzTensor::Tensor x_rsqrt = x_mean.rsqrt();
        EzTensor::Tensor output = x_rsqrt * x;
        return output;
    };

    Tensor RMSNormLayer::forward(Tensor& x){
       EzTensor::Tensor x_norm = _norm(x);
       EzTensor::Tensor output = x_norm * weights["input_layernorm_weight"];
       return output;
    };

}