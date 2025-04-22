#include <vector>

#pragma once

namespace EzTensor{
    class Tensor {
    public:
        float* data = nullptr;
        float* grad = nullptr;
        int size = 0;
        std::vector<int> shape;
        std::vector<int> strides;
        Tensor(std::vector<int>&input_shape);
        ~Tensor();
        void print_tensor();
        void reshape(std::vector<int>&input_shape);
        void transpose(int dim0, int dim1);
        void T();
        void fill_with(float value);
        void zero(); 
        void randn(float mean, float stddev);
        Tensor matmul(Tensor& rtensor);
        Tensor operator+(Tensor& rtensor);
        Tensor operator-(Tensor& rtensor);
        Tensor operator*(Tensor& rtensor);
        Tensor operator/(Tensor& rtensor);
        Tensor operator+(float value);
        Tensor operator-(float value);
        Tensor operator*(float value);
        Tensor operator/(float value);
        Tensor& operator+=(Tensor& rtensor);
        Tensor& operator-=(Tensor& rtensor);
        Tensor& operator*=(Tensor& rtensor);
        Tensor& operator/=(Tensor& rtensor);
    };
}
 

