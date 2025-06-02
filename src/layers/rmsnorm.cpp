#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "../include/layers.hpp"
#include "assert.h"

namespace EzTensor{


    RMSNormLayer::~RMSNormLayer(){
        weights.clear();
    }

    RMSNormLayer::RMSNormLayer(std::unordered_map<std::string,Tensor> input_weights, ModelArgs args){
        eps = 0.000001;
        weights = input_weights;
        std::vector<int> target_weight_shape = {1,1,args.dim};
        assert(weights["input_layernorm_weight"].shape == target_weight_shape);
    };

    Tensor RMSNormLayer::_norm(Tensor& x){
        EzTensor::Tensor x_pow = x.powr(2);
        EzTensor::Tensor x_mean = x_pow.mean(-1,true);
        x_mean += eps;
        EzTensor::Tensor x_rsqrt = x_mean.rsqrt();
        EzTensor::Tensor output = x * x_rsqrt;
        return output;
    };

    Tensor RMSNormLayer::forward(Tensor& x){
       EzTensor::Tensor x_norm = _norm(x);
       EzTensor::Tensor output = x_norm * weights["input_layernorm_weight"];
       return output;
    };

}