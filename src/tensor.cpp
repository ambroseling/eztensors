#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>  
#include <memory>
#include "include/tensor.hpp"
#include "kernels/matmul.cpp"
#include "kernels/outer.cpp"
#include "assert.h"

namespace EzTensor{

    //Constructor
    Tensor::Tensor(){shape={}; strides={}; size = 0;};
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

    bool is_broadcastable(std::vector<int>& shape_a, std::vector<int>& shape_b){
        if (shape_a.size() != shape_b.size()){
            return false;
        }
        for (int i=0; i<shape_a.size(); i++){

            if ((shape_a[i] != shape_b[i]) && (shape_a[i] != 1) && (shape_b[i] != 1)){
                return false;
            }
        }
        return true;
    }

    bool is_concatable(std::vector<int>& shape_a, std::vector<int>& shape_b, int dim){
        if (shape_a.size() != shape_b.size()){
            return false;
        }
        for (int i=0; i<shape_a.size(); i++){
            if (shape_a[i] != shape_b[i] && shape_a[i] != 1 && i != dim){
                return false;
            }
        }
        return true;
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

    std::vector<int> Tensor::compute_broadcast_strides(std::vector<int>&input_shape){
        std::vector<int> output_strides(input_shape.size(),0);
        int stride = 1;
        for (int dim = input_shape.size()-1; dim >= 0 ;dim--){
            if (input_shape[dim] == 1) output_strides[dim] = 0;
            else output_strides[dim] =  stride;
            if (dim != 0) stride  *= input_shape[dim];
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

    void Tensor::iterate_tensor_linear(std::function<void(int)> ops, ssize_t steps, ssize_t tensor_size){
        for (ssize_t offset = 0; offset < tensor_size; offset +=steps ){
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

    std::vector<int> compute_broadcast_shape(std::vector<int>& shape_a, std::vector<int>& shape_b)
    {
        std::vector<int> output_shape;
        assert(shape_a.size()==shape_b.size());
        for(int i=0;i<shape_a.size();i++){
            output_shape.push_back(std::max(shape_a[i],shape_b[i]));
        }
        return output_shape;
    }

    // Tensor Manipulations / Operations
    Tensor Tensor::outer(Tensor& rtensor, MM_MODE mode){
        if (shape.size() > 1 || rtensor.shape.size() > 1){
            throw std::runtime_error("Outer product of tensors with dimensions greater than 1 is not supported.");
        }
        std::vector<float> outer_product;
        switch(mode){
            case MM_MODE::SIMD: outer_product = kernel_outer_product_simd(data, rtensor.data); break;
            case MM_MODE::NAIVE: outer_product = kernel_outer_product_naive(data, rtensor.data); break;
            default: break;
        }
        std::vector<int> new_shape = {shape[0], rtensor.shape[0]};
        Tensor result(new_shape, outer_product);
        return result;
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


    void Tensor::print_tensor(bool display_data){
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

        if (display_data){
            std::cout<< "Data: " << std::endl;
            std::function<void(int)> print = [this](int offset){
               std::cout <<  (*this->data)[offset] << " " << std::endl;
            };
            int offset = 0; int dim = 0; 
            iterate_tensor(offset, dim, print);
        }
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
        iterate_tensor_linear(copyf , 1, result.size);
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
            iterate_tensor_linear(contigf, 1, result.size);
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
        iterate_tensor_linear(fill , 1, size);
    }

    void Tensor::randn(float mean, float stddev){
        std::random_device rd;  
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(mean,stddev);

        std::function<void(int)> gauss_draw = [this, &distribution, &generator](int offset){
            this->data->data()[offset] = distribution(generator);
        };
        iterate_tensor_linear(gauss_draw, 1, size);
    }

    Tensor Tensor::concat_with(Tensor& rtensor,int dim){
        bool is_cctable = is_concatable(shape,rtensor.shape, dim);
        if (!is_cctable)
            throw std::runtime_error("Unable to concatenate these 2 tensors.");
        int new_dim = shape[dim] +  rtensor.shape[dim];
        std::vector<int> new_shape = shape;
        new_shape[dim] = new_dim;
        Tensor result(new_shape);
        std::function<void(int)> concat_with_f = [this, &result, &rtensor](int offset){
            if (offset < size){
                result.data->data()[offset] =  this->data->data()[offset];
            }
            else{
                result.data->data()[offset] =  rtensor.data->data()[offset];
            }

        };
        iterate_tensor_linear(concat_with_f, 1, result.size);
        return result;
    }


    // Element Wise Opreations
    Tensor Tensor::operator+(Tensor& rtensor){
        // broadcasted tensor must be rtensor
        bool is_same_shape = shape == rtensor.shape;
        bool is_brdcastable = is_broadcastable(shape, rtensor.shape);
        if(!is_same_shape && !is_brdcastable){
            throw std::runtime_error("Tensors must have the same shape or be broadcastable!");
        }
        std::vector<int> new_shape = compute_broadcast_shape(shape,rtensor.shape);
        Tensor result(new_shape);
        std::vector<int> broadcast_stride = compute_broadcast_strides(rtensor.shape);
        std::function<void(int)> add_f = [this, &rtensor, &result, &broadcast_stride, &is_same_shape](int offset){
            int new_offset = offset;
            if (!is_same_shape){
                new_offset = compute_stride_offset(offset, result.strides,broadcast_stride);
            }
            result.data->data()[offset] = this->data->data()[offset]  + rtensor.data->data()[new_offset];
        };
        iterate_tensor_linear(add_f, 1, result.size);
        return result;
    };

    void Tensor::operator+=(float value){
        std::function<void(int)> add_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  + value;
        };
        iterate_tensor_linear(add_inplace, 1, size);
    };

    Tensor Tensor::operator-(Tensor& rtensor){
        // broadcasted tensor must be rtensor
        bool is_same_shape = shape == rtensor.shape;
        bool is_brdcastable = is_broadcastable(shape, rtensor.shape);
        if(!is_same_shape && !is_brdcastable){
            throw std::runtime_error("Tensors must have the same shape or be broadcastable!");
        }
        std::vector<int> new_shape = compute_broadcast_shape(shape,rtensor.shape);
        Tensor result(new_shape);
        std::vector<int> broadcast_stride = compute_broadcast_strides(rtensor.shape);
        std::function<void(int)> subtract_f = [this, &rtensor, &result, &broadcast_stride, &is_same_shape](int offset){
            int new_offset = offset;
            if (!is_same_shape){
                new_offset = compute_stride_offset(offset, result.strides,broadcast_stride);
            }
            result.data->data()[offset] = this->data->data()[offset]  - rtensor.data->data()[new_offset];
        };
        iterate_tensor_linear(subtract_f, 1, result.size);
        return result;
    };

    void Tensor::operator-=(float value){
        std::function<void(int)> subtract_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  - value;
        };
        iterate_tensor_linear(subtract_inplace, 1, size);
    };

    Tensor Tensor::operator*(Tensor& rtensor){
        // broadcasted tensor must be rtensor
        bool is_same_shape = shape == rtensor.shape;
        bool is_brdcastable = is_broadcastable(shape, rtensor.shape);
        if(!is_same_shape && !is_brdcastable){
            throw std::runtime_error("Tensors must have the same shape or be broadcastable!");
        }
        std::vector<int> new_shape = compute_broadcast_shape(shape,rtensor.shape);
        Tensor result(new_shape);
        std::vector<int> broadcast_stride = compute_broadcast_strides(rtensor.shape);
        std::function<void(int)> multiply_f = [this, &rtensor, &result, &broadcast_stride, &is_same_shape](int offset){
            int new_offset = offset;
            if (!is_same_shape){
                new_offset = compute_stride_offset(offset, result.strides,broadcast_stride);
            }
            result.data->data()[offset] = this->data->data()[offset]  * rtensor.data->data()[new_offset];
        };
        iterate_tensor_linear(multiply_f, 1, result.size);
        return result;
    };

    void Tensor::operator*=(float value){
        std::function<void(int)> multiply_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  * value;
        };
        iterate_tensor_linear(multiply_inplace, 1, size);
    };

    Tensor Tensor::operator/(Tensor& rtensor){
        // broadcasted tensor must be rtensor
        bool is_same_shape = shape == rtensor.shape;
        bool is_brdcastable = is_broadcastable(shape, rtensor.shape);
        if(!is_same_shape && !is_brdcastable){
            throw std::runtime_error("Tensors must have the same shape or be broadcastable!");
        }
        std::vector<int> new_shape = compute_broadcast_shape(shape,rtensor.shape);
        Tensor result(new_shape);
        std::vector<int> broadcast_stride = compute_broadcast_strides(rtensor.shape);
        std::function<void(int)> divide_f = [this, &rtensor, &result, &broadcast_stride, &is_same_shape](int offset){
            int new_offset = offset;
            if (!is_same_shape){
                new_offset = compute_stride_offset(offset, result.strides,broadcast_stride);
            }
            result.data->data()[offset] = this->data->data()[offset]  / rtensor.data->data()[new_offset];
        };
        iterate_tensor_linear(divide_f, 1, result.size);
        return result;
    };

    void Tensor::operator/=(float value){
        std::function<void(int)> divide_inplace = [this, &value](int offset){
            this->data->data()[offset] = this->data->data()[offset]  / value;
        };
        iterate_tensor_linear(divide_inplace, 1, size);
    };

    Tensor Tensor::operator+(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> add = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] + value;
        };
        iterate_tensor_linear(add, 1, result.size);
        return result;
    };

    Tensor Tensor::operator-(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> subtract = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] - value;
        };
        iterate_tensor_linear(subtract, 1, result.size);
        return result;
    };

    Tensor Tensor::operator*(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> multiply = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] * value;
        };
        iterate_tensor_linear(multiply, 1, result.size);
        return result;
    };

    Tensor Tensor::operator/(float value){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> divide = [this, value, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] / value;
        };
        iterate_tensor_linear(divide, 1, result.size);
        return result;
    };

    //Math Operations
    Tensor Tensor::complex_cos_sin(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> complex_cos_sin_f = [this, &result](int offset){
            if (offset % 2 == 0){
                result.data->data()[offset] = cos(this->data->data()[offset]);
            }
            else{
                result.data->data()[offset] = sin(this->data->data()[offset]);
            }   
        };
        iterate_tensor_linear(complex_cos_sin_f, 1, result.size); 
        return result;
    };

    Tensor Tensor::complex_sin_cos(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> complex_cos_sin_f = [this, &result](int offset){
            if (offset % 2 == 0){
                result.data->data()[offset] = sin(this->data->data()[offset]);
            }
            else{
                result.data->data()[offset] = cos(this->data->data()[offset]);
            }   
        };
        iterate_tensor_linear(complex_cos_sin_f, 1, result.size); 
        return result;
    };

    Tensor Tensor::last_dim_subtract(std::vector<int> input_shape){
        Tensor result(input_shape);

        std::function<void(int)> last_dim_subtract_f = [this, &result](int offset){
                result.data->data()[offset] = this->data->data()[offset /2] - this->data->data()[(offset/2)+1];
        };
        iterate_tensor_linear(last_dim_subtract_f, 1, result.size); 
        return result;
    }

    Tensor Tensor::rsqrt(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> sqrtff = [this, &result](int offset){
            result.data->data()[offset] = 1.0 / sqrtf(this->data->data()[offset]);
        };
        iterate_tensor_linear(sqrtff, 1, result.size); 
        return result;
    }

    Tensor Tensor::sqrt(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> sqrtff = [this, &result](int offset){
            result.data->data()[offset] = sqrtf(this->data->data()[offset]);
        };
        iterate_tensor_linear(sqrtff, 1, result.size); 
        return result;
    };


    Tensor Tensor::powr(float power){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> powf= [this, &result, &power](int offset){
            result.data->data()[offset] = pow(this->data->data()[offset], power);
        };
        iterate_tensor_linear(powf, 1, result.size);
        return result;
    };

    Tensor Tensor::rpowr(float base){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> powf= [this, &result, &base](int offset){
            result.data->data()[offset] = pow(base, this->data->data()[offset]);
        };
        iterate_tensor_linear(powf, 1, result.size);
        return result;
    };


    Tensor Tensor::sigmoid(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> sigmoidf= [this, &result](int offset){
            result.data->data()[offset] = 1 / (1  + exp(this->data->data()[offset]));
        };
        iterate_tensor_linear(sigmoidf, 1, result.size);
        return result;
    };

    Tensor Tensor::relu(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> reluf = [this, &result](int offset){
            result.data->data()[offset] = this->data->data()[offset] > 0 ? this->data->data()[offset]: 0;
        };
        iterate_tensor_linear(reluf, 1, result.size);
        return result;
    };

    Tensor Tensor::silu(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> reluf = [this, &result](int offset){
        result.data->data()[offset] = exp(this->data->data()[offset]) * (1 / (1  + exp(this->data->data()[offset])));
        };
        iterate_tensor_linear(reluf, 1, result.size);
        return result;
    };

    Tensor Tensor::htan(){
        Tensor result(shape);
        std::function<void(int)> tanhf = [this, &result](int offset){
            result.data->data()[offset] = tanh(this->data->data()[offset]);
        };
        iterate_tensor_linear(tanhf, 1, result.size);
        return result;
    };

    Tensor Tensor::sine(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> sinef = [this, &result](int offset){
            result.data->data()[offset] = sin(this->data->data()[offset]);
        };
        iterate_tensor_linear(sinef, 1, result.size);
        return result;
    };

    Tensor Tensor::cosine(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> cosinef = [this, &result](int offset){
            result.data->data()[offset] = cos(this->data->data()[offset]);
        };
        iterate_tensor_linear(cosinef, 1, result.size);
        return result;
    };

    Tensor Tensor::expo(){
        std::vector<int> new_shape = shape;
        Tensor result(new_shape);
        std::function<void(int)> expf = [this, &result](int offset){
            result.data->data()[offset] = exp(this->data->data()[offset]);
        };
        iterate_tensor_linear(expf, 1, result.size);
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
            iterate_tensor_linear(sum_all, 1, size);
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
            iterate_tensor_linear(sumf, 1, size);
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
            iterate_tensor_linear(mean_all, 1, size);
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
            iterate_tensor_linear(meanf, 1, size);
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
        iterate_tensor_linear(dropoutf, 1, result.size);
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
            iterate_tensor_linear(rep_memf, 1, result.size);
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

