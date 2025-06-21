#include <iostream>
#include "src/tensor.cpp"
#include "src/layers/attention.cpp"

int main(){
    EzTensor::ModelArgs args = EzTensor::ModelArgs();
    std::vector<int> shape = {1,2,3,4};
    EzTensor::Tensor dummy(shape);
    input_weights["dummy"] = dummy;
    EzTensor::AttentionLayer attn_layer = EzTensor::AttentionLayer(input_weights, args);    
    std::vector<int> xq_shape = {1,args.max_seq_len,args.n_heads * 64};
    std::vector<int> xk_shape = {1,args.max_seq_len,args.n_heads * 64};
    EzTensor::Tensor xq(xq_shape);
    EzTensor::Tensor xk(xk_shape);
    xq.fill_with(5.0);
    xk.fill_with(5.0);
    EzTensor::Tensor freqs = attn_layer.precompute_freq_cis(args.dim / args.n_heads, args.max_seq_len * 2, 10000.0);
    std::unordered_map<std::string,EzTensor::Tensor> rotary_embeddings = attn_layer.apply_rotary_embedding(xq,xk,freqs);
    
    return 0;
}