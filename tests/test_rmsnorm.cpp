#include <gtest/gtest.h>
#include "../src/include/constants.hpp"
#include "../src/tensor.cpp"
#include "../src/layers/rmsnorm.cpp"


TEST(AttentionTest,TestRMSNormLayer){
    std::vector<float> expected_data(1048576, 5.0);
    EzTensor::ModelArgs args;
    std::unordered_map<std::string,EzTensor::Tensor> input_weights;
    std::vector<int> weights_shape = {1,1,args.dim};
    EzTensor::Tensor norm_weights(weights_shape);
    norm_weights.fill_with(5.0);
    input_weights["input_layernorm_weight"] = norm_weights;
    std::vector<int> input_shape = {args.max_batch_size, args.max_seq_len, args.dim};
    EzTensor::Tensor input_tensor(input_shape);
    input_tensor.fill_with(2);
    EzTensor::RMSNormLayer norm_layer = EzTensor::RMSNormLayer(input_weights, args);
    EzTensor::Tensor norm_output = norm_layer.forward(input_tensor);
    for (int i=0; i<1048576 ;i++){
        float diff = (*norm_output.data)[i] - expected_data[i];
        EXPECT_NEAR(diff,0,0.0001);
    }
}