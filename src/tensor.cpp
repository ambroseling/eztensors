#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include "tensor.hpp"

// M x N
//ptr[col*M + row]


namespace EzTensor{

    int compute_size(std::vector<int>& shape){
        int num_elements = 1;
        for(const int& s : shape){
            num_elements *= s;
        }
        return num_elements;
    }

    void compute_strides(std::vector<int>& shape, std::vector<int>& strides){
        int stride = 1;
        for (int dim = shape.size()-1; dim >= 0 ;dim--){
            if (dim == shape.size()-1) strides[dim] = 1;
            else strides[dim] =  stride;
            if (dim != 0) stride  *= shape[dim];
        }
    }

    void iterate_tensor(int offset, int dim, std::vector<int>shape, std::vector<int>strides, std::function<void(int)> ops){
        if(dim == shape.size()){
            ops(offset);
            return;
        }
        for (int i=0; i< shape[dim]; i++){
            iterate_tensor(offset+i*strides[dim], dim+1, shape,strides,ops);
        }
    }

    Tensor::Tensor(std::vector<int>&input_shape):
        shape(std::move(input_shape)),
        strides(std::vector<int>(shape.size(),0))
        {
        compute_strides(shape,strides);
        size = compute_size(shape);
        data = (float*)malloc(sizeof(float)*size);
        memset(data,0,size);
    };

    Tensor::~Tensor(){
        free(this->data);
        free(this->grad);
    }

    // vibe coding

    void Tensor::print_tensor(){
        std::cout<< "Shape: [";
        for (int i=0;i<shape.size();i++){
            if (i < shape.size()-1) std::cout << shape[i] << ",";
            else std::cout << shape[i];
            
        }
        std::cout << "]" << std::endl;
        std::cout<< "Data: " << std::endl;
        std::function<void(int)> print = [this](int offset){
           std::cout <<  *(this->data + offset) << " " << std::endl;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, print);
    }

    void Tensor::zero(){
        memset(data,0,size);
    }

    void Tensor::reshape(std::vector<int>& input_shape){
        int input_size = compute_size(input_shape);
        if (input_size != size) {
            std::invalid_argument("Invalid reshape size!");
        }
        shape = std::move(input_shape);
        compute_strides(shape,strides);
    }

    void Tensor::transpose(int dim0, int dim1){
        if(dim0 > shape.size()-1 || dim1 > shape.size()-1){
            std::invalid_argument("Invalid dimensions for transpose!");
        }
        int temp = shape[dim0];
        shape[dim0] = shape[dim1];
        shape[dim1] = temp;
        compute_strides(shape,strides);
    }

    void Tensor::T(){
        if (shape.size() > 2){
           std::invalid_argument("Expected dimensions of 2!"); 
        }
        transpose(0,1);
    }

    void Tensor::fill_with(float value){
        std::function<void(int)> fill = [this,value](int offset){
            *(this->data + offset) =  value;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, fill);
    }

    void Tensor::randn(float mean, float stddev){
        std::random_device rd;  
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(mean,stddev);

        std::function<void(int)> gauss_draw = [this, &distribution, &generator](int offset){
            *(this->data + offset) = distribution(generator);
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, gauss_draw);
    }

    Tensor Tensor::operator+(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::invalid_argument("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> add = [this, &rtensor, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  + *(rtensor.data + offset);
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, add);
        return result;
    };

    Tensor Tensor::operator-(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::invalid_argument("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> subtract = [this, &rtensor, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  - *(rtensor.data + offset);
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, subtract);
        return result;
    };

    Tensor Tensor::operator*(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::invalid_argument("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> multiply = [this, &rtensor, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  * *(rtensor.data + offset);
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, multiply);
        return result;
    };

    Tensor Tensor::operator/(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::invalid_argument("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> divide = [this, &rtensor, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  / *(rtensor.data + offset);
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, divide);
        return result;
    };

    Tensor Tensor::operator+(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> add = [this, value, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  + value;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, add);
        return result;
    };

    Tensor Tensor::operator-(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> subtract = [this, value, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  - value;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, subtract);
        return result;
    };

    Tensor Tensor::operator*(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> multiply = [this, value, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  * value;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, multiply);
        return result;
    };

    Tensor Tensor::operator/(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> divide = [this, value, &result](int offset){
            *(result.data + offset) = *(this->data + offset)  / value;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, shape, strides, divide);
        return result;
    };

;}