#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "include/tensor.hpp"
#include "kernels/matmul.cpp"
#include "assert.h"

namespace EzTensor{

    //Constructor
    Tensor::Tensor(std::vector<int>&input_shape):
    shape(std::move(input_shape)),
    strides(std::vector<int>(shape.size(),0))
    {
    size = compute_size(shape);
    if (size == 0) {
        throw std::runtime_error("Cannot have tensor of size 0 or tensor with any dimension 0");
    }
    strides = compute_strides(shape); 
    data = std::make_shared<std::vector <float>>(size,0);
    };

    Tensor::Tensor(std::vector<int>&input_shape, std::shared_ptr<std::vector<float>> ref_data):
        shape(std::move(input_shape)),
        strides(std::vector<int>(shape.size(),0)),   
        data(ref_data){
        strides = compute_strides(shape); 
        size = compute_size(shape);
    }

    Tensor::Tensor(std::vector<int>& input_shape, std::vector<float>& input_data){
        int input_size = compute_size(input_shape);
        if (input_size != input_data.size()){
            throw std::runtime_error("Desired shape does not match shape of input data given");
        }
        size = input_size;
        data = std::make_shared<std::vector <float>>(size,0);
        *(this->data) = std::move(input_data);
        shape = std::move(input_shape);
        strides = compute_strides(shape); 
    }

    Tensor::~Tensor(){
        data.reset();
        grad.reset();
    }

    int Tensor::compute_size(std::vector<int>&input_shape){
        int num_elements = 1;
        for(const int& s : input_shape){
            num_elements *= s;
        }
        return num_elements;
    }

    std::vector<int> Tensor::compute_strides(std::vector<int>&input_shape){
        std::vector<int> output_strides(input_shape.size(),0);
        int stride = 1;
        for (int dim = shape.size()-1; dim >= 0 ;dim--){
            if (dim == shape.size()-1) output_strides[dim] = 1;
            else output_strides[dim] =  stride;
            if (dim != 0) stride  *= shape[dim];
        }
        return output_strides;
    }


    void Tensor::iterate_tensor(int offset, int dim, std::function<void(int)> ops){
        if(dim == shape.size()){
            ops(offset);
            return;
        }
        for (int i=0; i< shape[dim]; i++){
            iterate_tensor(offset+i*strides[dim], dim+1,ops);
        }
    }

    void Tensor::iterate_tensor_linear(std::function<void(int)> ops){
        for (ssize_t offset = 0; offset < this->size; offset ++ ){
            ops(offset);
        }
    }

    int Tensor::compute_stride_offset(int offset, std::vector<int>& src_stride, std::vector<int>& dest_strides){
        int new_offset = 0;
        for(int i=0 ; i<src_stride.size(); i++){
            int index = offset / src_stride[i];
            new_offset += index * dest_strides[i];
            offset %= src_stride[i];
        }
        return new_offset;
    }

    int Tensor::compute_reduce_offset(int offset, int dim, std::vector<int>& src_stride,  std::vector<int>& dest_stride){
        int new_offset = 0;
        for(int i=0 ; i<src_stride.size(); i++){
            int index = offset / src_stride[i];
            new_offset += i == dim ? 0 : index * dest_stride[i];
            offset %= src_stride[i];
        }
        return new_offset; 
    }

    int Tensor::compute_expand_offset(int offset, 
                                      std::vector<int>& new_strides, 
                                      std::vector<int>& old_shape ,
                                      std::vector<int>& new_shape){
        int new_offset = 0;
        for(int i=0 ; i<strides.size(); i++){
            int index = offset / strides[i];
            new_offset += old_shape[i] == new_shape[i]? index * new_strides[i]:0;
            offset %= strides[i];
        }
        return new_offset;
    }

    Tensor Tensor::matmul(Tensor& rtensor, MM_MODE mode){
        if (shape.size() != 2) {
            throw std::runtime_error( "Matrix Mulitplication of other sizes are not supported!" );
        }   
        if (shape[shape.size()-1] != rtensor.shape[0]){
            throw std::runtime_error( "Invalid Shapes for matrix multiplication" );
        }
        int M = shape[0], K = shape[shape.size()-1], N = rtensor.shape[rtensor.shape.size()-1];
        std::vector<int> new_shape = {M,N};
        Tensor result(new_shape);
        auto start = std::chrono::high_resolution_clock::now();
        switch(mode){
            case MM_MODE::SIMD: matmul_simd(data, rtensor.data, result.data, M, N, K); break;
            case MM_MODE::NAIVE: matmul_naive(data, rtensor.data, result.data, M, N, K); break;
            default: break;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double elapsed_seconds = elapsed.count() * 1e-9;
        std::cout << "Elapsed Seconds: " << elapsed_seconds << " sec" << std::endl;
        return result;
    }


    void Tensor::print_tensor(){
        std::cout<< "Shape: [";
        for (int i=0;i<shape.size();i++){
            if (i < shape.size()-1) std::cout << shape[i] << ",";
            else std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout<< "Stride: [";
        for (int i=0;i<strides.size();i++){
            if (i < strides.size()-1) std::cout << strides[i] << ",";
            else std::cout << strides[i];
        }
        std::cout << "]" << std::endl;
        std::cout<< "Data: " << std::endl;
        std::function<void(int)> print = [this](int offset){
           std::cout <<  (*this->data)[offset] << " " << std::endl;
        };
        int offset = 0; int dim = 0; 
        iterate_tensor(offset, dim, print);
    }

    void Tensor::zero(){
        memset(data.get(),0,size);
    }

    Tensor Tensor::copy(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> copyf = [this, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset];
        };
        iterate_tensor_linear(copyf);
        return result;
    }

    Tensor Tensor::contiguous(){
        if (this->is_contiguous()){
            return *(this);
        }
        else{
            std::vector<int> new_shape = shape;
            Tensor result(new_shape);
            std::function<void(int)> contigf = [this, &result](int offset){
                int non_contig_offset = compute_stride_offset(offset,result.strides,strides);
                result.data->data()[offset] = this->data->data()[non_contig_offset];
            };
            iterate_tensor_linear(contigf);
            return result;
        }
    }

    bool Tensor::is_contiguous(){
        if (strides[strides.size()-1] != 1){
            return false;
        }
        return true;
    }


    Tensor Tensor::view(std::vector<int>& target_shape){
        if (!this->is_contiguous()){ 
            throw std::runtime_error( "Cannot return a view for a non-contiguous tensor");
        }
        int input_size = compute_size(target_shape);
        std::vector<int> result_shape = target_shape;
        Tensor result(result_shape, data);

        std::cout << "Size : " << size << std::endl;
        std::cout << "Input size :" << input_size << std::endl;

        if (input_size != size) { 
            throw std::runtime_error( "Invalid shapes for returning a view of this tensor with target shape" );
        }
        return result;
    }


    Tensor Tensor::reshape(std::vector<int>& input_shape){
        int input_size = compute_size(input_shape);
        if (input_size != size) {
            throw std::runtime_error("Invalid reshape size!");
        }
        if(this->is_contiguous()){
           Tensor result = this->view(input_shape);
           return result;
        }
        else{
            Tensor contig_t = this->contiguous();
            Tensor result = contig_t.view(input_shape);
            return result;
        }
    }

    Tensor Tensor::transpose(int dim0, int dim1){
        if(dim0 > shape.size()-1 || dim1 > shape.size()-1 || dim0 < 0 || dim1 < 0){
            throw std::runtime_error("Invalid dimensions for transpose!");
        }
        Tensor result(shape, data);
        std::swap(result.shape[dim0],result.shape[dim1]);
        std::swap(result.strides[dim0],result.strides[dim1]);
        return result;
    }

    Tensor Tensor::T(){
        if (shape.size() > 2){
           throw std::runtime_error("Transposition only works on tensor with 2 dimensions!"); 
        }
        Tensor t = transpose(0,1);
        return t;
    }

    void Tensor::fill_with(float value){
        std::function<void(int)> fill = [this,value](int offset){
            this->data->data()[offset] =  value;
        };
        iterate_tensor_linear(fill);
    }

    void Tensor::randn(float mean, float stddev){
        std::random_device rd;  
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(mean,stddev);

        std::function<void(int)> gauss_draw = [this, &distribution, &generator](int offset){
            this->data->data()[offset] = distribution(generator);
        };
        iterate_tensor_linear(gauss_draw);
    }

    Tensor Tensor::operator+(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::runtime_error("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> add = [this, &rtensor, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset]  + rtensor.data->data()[offset];
        };
        iterate_tensor_linear(add);
        return result;
    };

    void Tensor::operator+=(float value){
        std::function<void(int)> add_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  + value;
        };
        iterate_tensor_linear(add_inplace);
    };

    Tensor Tensor::operator-(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::runtime_error("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> subtract = [this, &rtensor, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset]  - rtensor.data->data()[offset];
        };
        iterate_tensor_linear(subtract);
        return result;
    };

    void Tensor::operator-=(float value){
        std::function<void(int)> subtract_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  - value;
        };
        iterate_tensor_linear(subtract_inplace);
    };

    Tensor Tensor::operator*(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::runtime_error("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> multiply = [this, &rtensor, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset]  * rtensor.data->data()[offset];
        };
        iterate_tensor_linear(multiply);
        return result;
    };

    void Tensor::operator*=(float value){
        std::function<void(int)> multiply_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  * value;
        };
        iterate_tensor_linear(multiply_inplace);
    };

    Tensor Tensor::operator/(Tensor& rtensor){
        if(shape != rtensor.shape){
            throw std::runtime_error("Tensors must have the same shape!");
        }
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> divide = [this, &rtensor, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset]  / rtensor.data->data()[offset];
        };
        iterate_tensor_linear(divide);
        return result;
    };

    void Tensor::operator/=(float value){
        std::function<void(int)> divide_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  / value;
        };
        iterate_tensor_linear(divide_inplace);
    };

    Tensor Tensor::operator+(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> add = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] + value;
        };
        iterate_tensor_linear(add);
        return result;
    };

    Tensor Tensor::operator-(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> subtract = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] - value;
        };
        iterate_tensor_linear(subtract);
        return result;
    };

    Tensor Tensor::operator*(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> multiply = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] * value;
        };
        iterate_tensor_linear(multiply);
        return result;
    };

    Tensor Tensor::operator/(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> divide = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] / value;
        };
        iterate_tensor_linear(divide);
        return result;
    };

    //Math Operations
    Tensor Tensor::rsqrt(){
        Tensor result(shape);
        std::function<void(int)> sqrtff = [this, &result](int offset){
            result.data->data()[offset] = 1.0 / sqrtf(this->data->data()[offset]);
        };
        iterate_tensor_linear(sqrtff); 
        return result;
    }

    Tensor Tensor::sqrt(){
        Tensor result(shape);
        std::function<void(int)> sqrtff = [this, &result](int offset){
            result.data->data()[offset] = sqrtf(this->data->data()[offset]);
        };
        iterate_tensor_linear(sqrtff); 
        return result;
    };


    Tensor Tensor::powr(float power){
        Tensor result(shape);
        std::function<void(int)> powf= [this, &result, &power](int offset){
            result.data->data()[offset] = pow(this->data->data()[offset], power);
        };
        iterate_tensor_linear(powf);
        return result;
    };


    Tensor Tensor::sigmoid(){
        Tensor result(shape);
        std::function<void(int)> sigmoidf= [this, &result](int offset){
            result.data->data()[offset] = 1 / (1  + exp(this->data->data()[offset]));
        };
        iterate_tensor_linear(sigmoidf);
        return result;
    };

    Tensor Tensor::relu(){
        Tensor result(shape);
        std::function<void(int)> reluf = [this, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] > 0 ? this->data->data()[offset]: 0;
        };
        iterate_tensor_linear(reluf);
        return result;
    };

    Tensor Tensor::silu(){
        Tensor result(shape);
        std::function<void(int)> reluf = [this, &result](int offset){
            result.data->data()[offset] = exp(this->data->data()[offset]) * (1 / (1  + exp(this->data->data()[offset])));
        };
        iterate_tensor_linear(reluf);
        return result;
    };

    Tensor Tensor::htan(){
        Tensor result(shape);
        std::function<void(int)> tanhf = [this, &result](int offset){
            result.data->data()[offset] = tanh(this->data->data()[offset]);
        };
        iterate_tensor_linear(tanhf);
        return result;
    };

    Tensor Tensor::sine(){
        Tensor result(shape);
        std::function<void(int)> sinef = [this, &result](int offset){
            result.data->data()[offset] = sin(this->data->data()[offset]);
        };
        iterate_tensor_linear(sinef);
        return result;
    };

    Tensor Tensor::cosine(){
        Tensor result(shape);
        std::function<void(int)> cosinef = [this, &result](int offset){
            result.data->data()[offset] = cos(this->data->data()[offset]);
        };
        iterate_tensor_linear(cosinef);
        return result;
    };

    Tensor Tensor::expo(){
        Tensor result(shape);
        std::function<void(int)> expf = [this, &result](int offset){
            result.data->data()[offset] = exp(this->data->data()[offset]);
        };
        iterate_tensor_linear(expf);
        return result;
    };

    Tensor Tensor::sum(int dim, bool keepdim){

        if (dim == INT_MIN) {
            std::vector<int> new_shape = {1};
            Tensor result(new_shape);
            float summation = 0.0f;
            std::function<void(int)> sum_all = [this, &summation](int offset){
                summation += this->data->data()[offset];
            };
            iterate_tensor_linear(sum_all);
            result.data->data()[0] = summation;    
            return result;
        }
        else{
            if (dim != -1 && (dim > shape.size()-1 || dim < 0)){
                throw std::runtime_error("Invalid reduction dimension");
            }
            dim = dim == -1 ? shape.size()-1 : dim;
            std::vector<int> new_shape;
            if (shape.size() == 1) {
                new_shape = {1};
            }
            else {
                new_shape = shape; 
                if(keepdim)
                    new_shape[dim] = 1;
                else
                    new_shape.erase(new_shape.begin()+dim);
            }
            Tensor result(new_shape);
            std::vector<int> reduc_stride = result.strides;
            if(!keepdim)
                reduc_stride.insert(reduc_stride.begin()+dim,0);
            std::function<void(int)> sumf = [this, &dim, &result, &reduc_stride](int offset){
                int new_offset = compute_reduce_offset(offset, dim, strides, reduc_stride);
                result.data->data()[new_offset] += this->data->data()[offset];
            };
            iterate_tensor_linear(sumf);
            return result;
        }
    };


    Tensor Tensor::mean(int dim, bool keepdim){

        if (dim == INT_MIN) {
            std::vector<int> new_shape = {1};
            Tensor result(new_shape);
            float mean = 0.0f;
            std::function<void(int)> mean_all = [this, &mean](int offset){
                mean += this->data->data()[offset];
            };
            iterate_tensor_linear(mean_all);
            result.data->data()[0] = mean * (1.0 / size);    
            return result;
        }
        else{
            if (dim != -1 && (dim > shape.size()-1 || dim < 0)){
                throw std::runtime_error("Invalid reduction dimension");
            }
            dim = dim == -1 ? shape.size()-1 : dim;
            std::vector<int> new_shape;
            if (shape.size() == 1) {
                new_shape = {1};
            }
            else {
                new_shape = shape; 
                if(keepdim)
                    new_shape[dim] = 1;
                else
                    new_shape.erase(new_shape.begin()+dim);
            }
            Tensor result(new_shape);
            std::vector<int> reduc_stride = result.strides;
            if(!keepdim)
                reduc_stride.insert(reduc_stride.begin()+dim,0);
            std::function<void(int)> meanf = [this, &dim, &result, &reduc_stride](int offset){
                int new_offset = compute_reduce_offset(offset, dim, strides, reduc_stride);
                result.data->data()[new_offset] += this->data->data()[offset] * (1.0 / shape[dim]);
            };
            iterate_tensor_linear(meanf);
            return result;
        }
    };

    Tensor Tensor::dropout(float prob){
        std::vector<int> new_shape = shape; 
        Tensor result(new_shape);
        std::function<void(int)> dropoutf = [this, &result, &prob](int offset){
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            result.data->data()[offset] = r <= prob? this->data->data()[offset] : 0.0f;
        };
        iterate_tensor_linear(dropoutf);
        return result;
    }

    Tensor Tensor::expand(std::vector<int> target_shape, bool rep_mem){
        bool broadcastable = false;
        if (target_shape == shape)
            return *(this);
        if (target_shape.size() != shape.size())
            throw std::runtime_error("Target broadcast shape is smaller than existing shape!");
        int i = shape.size()-1; 
        while(i >= 0){
            if (shape[i] == target_shape[i] || shape[i] == 1){
                i--;
            }  
            else {    
                throw std::runtime_error("The size at dimension " + std::to_string(i) + " must match or the input size must be 1.");
            }
        }
        std::vector<int> new_shape = target_shape;
        if (rep_mem){
            Tensor result(new_shape);
            std::function<void(int)> rep_memf = [this, &result, &target_shape](int offset){
                int new_offset = compute_expand_offset(offset, result.strides, shape, target_shape);
                result.data->data()[offset] = this->data->data()[new_offset];
            };
            iterate_tensor_linear(rep_memf);
            return result;
        }
        else{
            Tensor result(new_shape,data);
            return result;
        }
    }
    Tensor Tensor::softmax(int dim){
        Tensor new_shape = shape;
        Tensor exp_tensor = this->expo();
        // Tensor exp_sum_tensor = exp_tensor.sum(dim);
        Tensor result  = exp_tensor;
        return result;
    }
}

