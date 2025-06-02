#include <vector>

#pragma once

namespace EzTensor{

    enum class MM_MODE { SIMD, NAIVE };

    class Tensor {
    public:
        // Attributes
        ssize_t size = 0;
        std::shared_ptr<std::vector<float>> data;
        std::shared_ptr<std::vector<float>> grad;
        std::vector<int> shape;
        std::vector<int> strides;

        //Helper Functions
        int compute_stride_offset(int offset, std::vector<int>& src_stride, std::vector<int>& dest_strides);
        int compute_reduce_offset(int offset, int dim, std::vector<int>& src_stride,  std::vector<int>& dest_stride);
        int compute_expand_offset(int offset, std::vector<int>& new_strides, std::vector<int>& old_shape ,std::vector<int>& new_shape);
        int compute_size(std::vector<int>& input_shape);
        std::vector<int> compute_strides(std::vector<int>& input_shape);
        std::vector<int> compute_broadcast_strides(std::vector<int>&input_shape);

        void iterate_tensor(int offset, int dim,  std::function<void(int)> ops);
        void iterate_tensor_linear(std::function<void(int)> ops, ssize_t steps, ssize_t tensor_size);
        void print_tensor(bool display_data);

        //Constructors
        Tensor();
        Tensor(std::vector<int>& input_shape); //✔
        Tensor(std::vector<int>& input_shape, std::shared_ptr<std::vector<float>> data); //✔
        Tensor(std::vector<int>& input_shape, std::vector<float>& input_data);
        
        //Destructor
        ~Tensor();

        //Tensor Manipulations
        bool is_contiguous(); //✔
        Tensor contiguous(); //✔
        Tensor copy(); //✔
        Tensor reshape(std::vector<int>&input_shape); //✔
        void squeeze(int dim);
        void unsqueeze(int dim);
        Tensor transpose(int dim0, int dim1); //✔
        Tensor T(); //✔
        void fill_with(float value); //✔ 🇹
        void zero(); //✔
        void randn(float mean, float stddev); //✔
        Tensor expand(std::vector<int> target_shape, bool rep_mem);
        Tensor view(std::vector<int>& target_shape);
        Tensor matmul(Tensor& rtensor, MM_MODE mode); //✔
        Tensor outer(Tensor& rtensor, MM_MODE mode);
        Tensor concat_with(Tensor& rtensor,int dim);

        //Element-Wise Operations
        Tensor operator+(Tensor& rtensor); //✔ 🇹
        Tensor operator-(Tensor& rtensor); //✔ 🇹
        Tensor operator*(Tensor& rtensor); //✔
        Tensor operator/(Tensor& rtensor); //✔
        Tensor operator+(float value); //✔ 🇹 
        Tensor operator-(float value); //✔
        Tensor operator*(float value); //✔
        Tensor operator/(float value); //✔
        void operator+=(float value); //✔ 🇹
        void operator-=(float value); //✔ 🇹
        void operator*=(float value); //✔
        void operator/=(float value); //✔
        Tensor sigmoid(); //✔ 
        Tensor silu(); //✔ 
        Tensor relu(); //✔ 
        Tensor sine(); //✔ 
        Tensor cosine(); // ✔ 
        Tensor htan(); // ✔ 
        Tensor expo(); // ✔ 
        Tensor sqrt(); // ✔ 
        Tensor rsqrt();
        Tensor powr(float power); // t^power ele-wise
        Tensor rpowr(float base); // base^t ele-wise
        Tensor dropout(float prob);
        Tensor complex_cos_sin(); // special rope op
        Tensor complex_sin_cos();
        Tensor last_dim_subtract(std::vector<int> input_shape);

        //Reductions
        Tensor sum(int dim, bool keepdim); //✔
        Tensor mean(int dim, bool keepdim); //✔
        Tensor var(int dim, bool keepdim); //✔
        Tensor softmax(int dim); 
        Tensor layernorm(float eps);
        Tensor rmsnorm(float eps);

        // Loss Functions
        Tensor bceloss(Tensor& rtensor);
        Tensor celoss(Tensor& rtensor);
    };
}
 

