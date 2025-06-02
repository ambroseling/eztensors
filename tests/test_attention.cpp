#include <gtest/gtest.h>
#include "../src/include/constants.hpp"
#include "../src/tensor.cpp"
#include "../src/layers/attention.cpp"


TEST(AttentionTest, TestPreComputeFreqCis) {
    EzTensor::ModelArgs args = EzTensor::ModelArgs();
    std::unordered_map<std::string,EzTensor::Tensor> input_weights;
    std::vector<int> shape = {1,2,3,4};
    EzTensor::Tensor dummy(shape);
    input_weights["dummy"] = dummy;
    EzTensor::AttentionLayer attn_layer = EzTensor::AttentionLayer(input_weights, args);
    EzTensor::Tensor freqs = attn_layer.precompute_freq_cis(args.dim / args.n_heads, args.max_seq_len, 10000.0);

}
