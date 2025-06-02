#include <iostream>
#include "src/tensor.cpp"
// #include "src/layers/attention.cpp"
#include "src/layers/feedforward.cpp"

int main(){
    EzTensor::ModelArgs args;
    std::unordered_map<std::string,EzTensor::Tensor> input_weights;
    std::vector<int> down_proj_weight_shape = {2048,8192};
    EzTensor::Tensor down_proj_weight(down_proj_weight_shape);
    down_proj_weight.fill_with(5.0);
    std::vector<int> gate_proj_weight_shape = {8192,2048};
    EzTensor::Tensor gate_proj_weight(gate_proj_weight_shape);
    gate_proj_weight.fill_with(5.0);
    std::vector<int> up_proj_weight_shape = {8192,2048};
    EzTensor::Tensor up_proj_weight(up_proj_weight_shape);
    up_proj_weight.fill_with(5.0);
    input_weights["mlp_down_proj_weight"] = down_proj_weight;
    input_weights["mlp_gate_proj_weight"] = gate_proj_weight;
    input_weights["mlp_up_proj_weight"] = up_proj_weight;

    std::vector<int> input_shape = {args.max_seq_len, args.dim};
    EzTensor::Tensor input_tensor(input_shape);
    input_tensor.fill_with(2);
    EzTensor::FeedForwardLayer ff_layer = EzTensor::FeedForwardLayer(input_weights, args);
    EzTensor::Tensor output = ff_layer.forward(input_tensor);
    return 0;
}