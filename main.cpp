#include <iostream>
#include "src/tensor.cpp"

int main(){
    // std::vector<int> shape_1 = {2,2};
    // EzTensor::Tensor t1(shape_1);
    // t1.fill_with(5);

    // std::vector<int> shape_2 = {2,2};
    // EzTensor::Tensor t2(shape_2);
    // t2.fill_with(2);

    // EzTensor::Tensor t3 = t1 + t2;
    // t3.print_tensor();

    // EzTensor::Tensor t4 = t3 / t2;
    // t4.print_tensor();


    // EzTensor::Tensor t5 = t4 * t1;
    // t5.print_tensor();

    // EzTensor::Tensor t6 = t4 * 3.14;
    // t6.print_tensor();

    // EzTensor::Tensor t7 = t4 / 2.123;
    // t7.print_tensor();

    // std::vector<int> shape_t = {2,3};
    // EzTensor::Tensor t_transpose(shape_t);
    // t_transpose.transpose(0,1);
    // t_transpose.print_tensor();
    // t_transpose.fill_with(5.3);
    // t_transpose.print_tensor();
    // t_transpose.zero();
    // t_transpose.print_tensor();
    
    std::vector<int> rand_shape = {4,3};
    EzTensor::Tensor randn_t(rand_shape);  
    randn_t.randn(0.0f,1.0f);
    randn_t.print_tensor();
    return 0;
}