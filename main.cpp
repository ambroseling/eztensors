#include <iostream>
#include "src/tensor.cpp"

int main(){
    std::vector<float> expected_data = {12.5};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(INT_MIN, false);
    mean.print_tensor();
    return 0;
}