#include <gtest/gtest.h>
#include "../src/tensor.cpp"

TEST(TensorTest, TestCreationDefinedShape) {
    std::vector<int> shape = {2,3,4};
    std::vector<int> expected_shape = {2,3,4};
    std::vector<int> expected_stride = {12,4,1};
    std::vector<float> zeros(24,0.0);
    EzTensor::Tensor t(shape);
    EXPECT_EQ(*(t.data),zeros);
    EXPECT_EQ(t.shape,expected_shape);
    EXPECT_EQ(t.strides, expected_stride);
    EXPECT_EQ(t.size, 24);
}

TEST(TensorTest, TestCreationZeroShape){
    std::vector<int> input_shape = {0};
    EXPECT_THROW(
    {
    try {
        EzTensor::Tensor t(input_shape);
    }
    catch(std::runtime_error& exception){
        EXPECT_STREQ("Cannot have tensor of size 0 or tensor with any dimension 0", exception.what() );
        throw;
    }
    }, std::runtime_error);
}

TEST(TensorTest, TestCreationShared){
    std::vector<float> fives(24,5.0);
    std::vector<int> input_shape = {2,3,4};
    EzTensor::Tensor t(input_shape);
    t.fill_with(5.0);
    std::vector<int> input_shape_s = {2,3,4};
    EzTensor::Tensor ts(input_shape_s, t.data);
    EXPECT_EQ(*(ts.data),fives);
}

TEST(TensorTest, TestCreationInputData){
    std::vector<int> input_shape = {3,2};
    std::vector<float> expected_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    std::vector<float> input_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    EzTensor::Tensor t(input_shape, input_data);
    EXPECT_EQ(*(t.data),expected_data);
}

TEST(TensorTest, TestCreationInputDataWrongShape){
    std::vector<int> input_shape = {3,3};
    std::vector<float> expected_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    std::vector<float> input_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    EXPECT_THROW(
        {
        try {
            EzTensor::Tensor t(input_shape, input_data);
        }
        catch(std::runtime_error& exception){
            EXPECT_STREQ("Desired shape does not match shape of input data given", exception.what() );
            throw;
        }
        }, std::runtime_error);
    }

TEST(TensorTest, TestFillWith) {
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {5,5,5,5};
    EzTensor::Tensor t(input_shape);
    t.fill_with(5.0);
    EXPECT_EQ(*(t.data),expected_data);
}

TEST(TensorTest, TestFillWithScalar){
    std::vector<int> input_shape = {1};
    std::vector<float> expected_data = {13.0};
    EzTensor::Tensor t(input_shape);
    t.fill_with(13.0);
    EXPECT_EQ(*(t.data),expected_data);
}

TEST(TensorTest, TestAddWrongShape){
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5);
    std::vector<int> shape_b = {2,3};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3);
    EXPECT_THROW(
    {
    try {
        EzTensor::Tensor tc = ta + tb;
    }
    catch(std::runtime_error& exception){
        EXPECT_STREQ("Tensors must have the same shape!", exception.what() );
        throw;
    }
    }, std::runtime_error);
}

TEST(TensorTest, TestAddDiffTensor){
    std::vector<float> expected_data = {8.0,8.0,8.0,8.0};
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5.0);
    std::vector<int> shape_b = {2,2};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3.0);
    EzTensor::Tensor tc = ta + tb;
    EXPECT_EQ(*(tc.data),expected_data);
}
TEST(TensorTest, TestAddSameTensor){
    std::vector<float> expected_data = {24.0,24.0,24.0,24.0};
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(12.0);
    EzTensor::Tensor tb = ta + ta;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestAddFloat){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {2025.052,2025.052,2025.052,2025.052};
    EzTensor::Tensor ta(input_shape);
    ta.fill_with(23.0);
    EzTensor::Tensor tb = ta + 2002.052;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestAddInplace){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {2025.052,2025.052,2025.052,2025.052};
    EzTensor::Tensor t(input_shape);
    t.fill_with(23.0);
    t += 2002.052;
    EXPECT_EQ(*(t.data),expected_data);
}




// SUBTRACT
TEST(TensorTest, TestSubtractWrongShape){
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5);
    std::vector<int> shape_b = {2,3};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3);
    EXPECT_THROW(
    {
    try {
        EzTensor::Tensor tc = ta - tb;
    }
    catch(std::runtime_error& exception){
        EXPECT_STREQ("Tensors must have the same shape!", exception.what() );
        throw;
    }
    }, std::runtime_error);
}

TEST(TensorTest, TestSubtractDiffTensor){
    std::vector<float> expected_data = {2.0,2.0,2.0,2.0};
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5.0);
    std::vector<int> shape_b = {2,2};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3.0);
    EzTensor::Tensor tc = ta - tb;
    EXPECT_EQ(*(tc.data),expected_data);
}

TEST(TensorTest, TestSubtractSameTensor){
    std::vector<float> expected_data(4,0);
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(12.0);
    EzTensor::Tensor tb = ta - ta;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestSubtractFloat){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {2002.052,2002.052,2002.052,2002.052};
    EzTensor::Tensor ta(input_shape);
    ta.fill_with(2025.052);
    EzTensor::Tensor tb = ta - 23.0;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestSubtractInplace){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {2002.052,2002.052,2002.052,2002.052};
    EzTensor::Tensor t(input_shape);
    t.fill_with(2025.052);
    t -= 23.0;
    EXPECT_EQ(*(t.data),expected_data);
}

// MULTIPLY
TEST(TensorTest, TestMulitplyWrongShape){
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5);
    std::vector<int> shape_b = {2,3};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3);
    EXPECT_THROW(
    {
    try {
        EzTensor::Tensor tc = ta * tb;
    }
    catch(std::runtime_error& exception){
        EXPECT_STREQ("Tensors must have the same shape!", exception.what() );
        throw;
    }
    }, std::runtime_error);
}

TEST(TensorTest, TestMulitplyDiffTensor){
    std::vector<float> expected_data = {15.0,15.0,15.0,15.0};
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5.0);
    std::vector<int> shape_b = {2,2};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3.0);
    EzTensor::Tensor tc = ta * tb;
    EXPECT_EQ(*(tc.data),expected_data);
}

TEST(TensorTest, TestMulitplySameTensor){
    std::vector<float> expected_data(4,144.0);
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(12.0);
    EzTensor::Tensor tb = ta * ta;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestMulitplyFloat){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {156.0,156.0,156.0,156.0};
    EzTensor::Tensor ta(input_shape);
    ta.fill_with(12.0);
    EzTensor::Tensor tb = ta * 13.0;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestMulitplyInplace){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {156.0,156.0,156.0,156.0};
    EzTensor::Tensor t(input_shape);
    t.fill_with(12.0);
    t *= 13.0;
    EXPECT_EQ(*(t.data),expected_data);
}

// DIVIDE
TEST(TensorTest, TestDivideWrongShape){
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(5);
    std::vector<int> shape_b = {2,3};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3);
    EXPECT_THROW(
    {
    try {
        EzTensor::Tensor tc = ta / tb;
    }
    catch(std::runtime_error& exception){
        EXPECT_STREQ("Tensors must have the same shape!", exception.what() );
        throw;
    }
    }, std::runtime_error);
}

TEST(TensorTest, TestDivideDiffTensor){
    std::vector<float> expected_data = {5.0,5.0,5.0,5.0};
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(15.0);
    std::vector<int> shape_b = {2,2};
    EzTensor::Tensor tb(shape_b);
    tb.fill_with(3.0);
    EzTensor::Tensor tc = ta / tb;
    EXPECT_EQ(*(tc.data),expected_data);
}

TEST(TensorTest, TestDivideSameTensor){
    std::vector<float> expected_data(4,1.0);
    std::vector<int> shape_a = {2,2};
    EzTensor::Tensor ta(shape_a);
    ta.fill_with(12.0);
    EzTensor::Tensor tb = ta / ta;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestDivideFloat){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {12.0,12.0,12.0,12.0};
    EzTensor::Tensor ta(input_shape);
    ta.fill_with(24.0);
    EzTensor::Tensor tb = ta / 2.0;
    EXPECT_EQ(*(tb.data),expected_data);
}

TEST(TensorTest, TestDivideInplace){
    std::vector<int> input_shape = {2,2};
    std::vector<float> expected_data = {12.0,12.0,12.0,12.0};
    EzTensor::Tensor t(input_shape);
    t.fill_with(24.0);
    t /= 2.0;
    EXPECT_EQ(*(t.data),expected_data);
}



TEST(TensorTest, TestView){
    std::vector<float> expected_data = {2.0,4.0,6.0,8.0,10.0,12.0};
    std::vector<int> input_shape = {6};
    std::vector<float> input_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    EzTensor::Tensor t(input_shape, input_data);
    std::vector<int> target_shape = {2,3};
    EzTensor::Tensor tv = t.view(target_shape);
    std::vector<int> input_shape_r = {2,3};
    std::vector<float> input_data_r = {1.0,2.0,3.0,4.0,5.0,6.0};
    EzTensor::Tensor tr(input_shape_r, input_data_r);
    EzTensor::Tensor td = tv + tr;
    EXPECT_EQ(*(td.data), expected_data);
}

TEST(TensorTest, TestViewNonContig){
    std::vector<int> input_shape = {2,3};
    std::vector<int> target_shape = {2,2,2};
    std::vector<float> input_data = {1.0,2.0,3.0,4.0,5.0,6.0};
    EzTensor::Tensor t(input_shape, input_data);
    EzTensor::Tensor t_trans = t.T(); // shape: 3,2
    EXPECT_THROW(
        {
        try {
            EzTensor::Tensor tv = t_trans.view(target_shape);  
        }
        catch(std::runtime_error& exception){
            EXPECT_STREQ("Cannot return a view for a non-contiguous tensor", exception.what() );
            throw;
        }
        }, std::runtime_error);
}


TEST(TensorTest, TestTransposeContig){
    std::vector<float> expected_data = {1.0,  5.0,  9.0,  2.0,  6.0, 10.0,  3.0,  7.0, 11.0,  4.0,  8.0, 12.0, 13.0, 17.0, 21.0, 14.0, 18.0, 22.0, 15.0, 19.0, 23.0, 16.0, 20.0, 24.0};
    std::vector<int> input_shape = {2,3,4};
    std::vector<int> target_shape = {12,2};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data);
    EzTensor::Tensor t_trans = t.transpose(1,2);
    EzTensor::Tensor t_trans_contig = t_trans.contiguous();
    EzTensor::Tensor tv = t_trans_contig.view(target_shape);
    EXPECT_EQ(*(tv.data),expected_data);
}

// Sum operation
TEST(TensorTest, TestSumAllReduceDontKeep){
    std::vector<float> expected_data = {300.0};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(INT_MIN, false);
    EXPECT_EQ(*(sum.data), expected_data);
}

TEST(TensorTest, TestSumAllReduceKeep){
    std::vector<float> expected_data = {300.0};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(INT_MIN, true);
    EXPECT_EQ(*(sum.data), expected_data);
}


TEST(TensorTest, TestSumNDimReduceDontKeep){
    std::vector<float> expected_data = {15,18,21,24,51,54,57,60};
    std::vector<int> expect_shape = {2,4};
    std::vector<int> expect_strides = {4,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(1,false);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}

TEST(TensorTest, TestSumNDimReduceKeep){
    std::vector<float> expected_data = {15,18,21,24,51,54,57,60};
    std::vector<int> expect_shape = {2,1,4};
    std::vector<int> expect_strides = {4,4,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(1,true);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}

TEST(TensorTest, TestSumLastDimReduceDontKeep){
    std::vector<float> expected_data = {10,26,42,58,74,90};
    std::vector<int> expect_shape = {2,3};
    std::vector<int> expect_strides = {3,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(-1,false);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}

TEST(TensorTest, TestSumLastDimReduceKeep){
    std::vector<float> expected_data = {10,26,42,58,74,90};
    std::vector<int> expect_shape = {2,3,1};
    std::vector<int> expect_strides = {3,1,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(-1,true);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}

TEST(TensorTest, TestSumOneDimReduceDontKeep){
    std::vector<float> expected_data = {55.0};
    std::vector<int> expect_shape = {1};
    std::vector<int> expect_strides = {1};
    std::vector<int> input_shape = {10};
    std::vector<float> input_data = {1,2,3,4,5,6,7,8,9,10};
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(0, false);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}

TEST(TensorTest, TestSumOneDimReduceKeep){
    std::vector<float> expected_data = {55.0};
    std::vector<int> expect_shape = {1};
    std::vector<int> expect_strides = {1};
    std::vector<int> input_shape = {10};
    std::vector<float> input_data = {1,2,3,4,5,6,7,8,9,10};
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor sum = t.sum(0, true);
    EXPECT_EQ(*(sum.data), expected_data);
    EXPECT_EQ(sum.shape, expect_shape);
    EXPECT_EQ(sum.strides,expect_strides);
}


TEST(TensorTest, TestSumInvalidDim){
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EXPECT_THROW(
        {
        try {
            EzTensor::Tensor sum = t.sum(3,false);
        }
        catch(std::runtime_error& exception){
            EXPECT_STREQ("Invalid reduction dimension", exception.what() );
            throw;
        }
        }, std::runtime_error);
}


// Mean Operation
TEST(TensorTest, TestMeanAllReduceDontKeep){
    std::vector<float> expected_data = {12.5};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(INT_MIN, false);
    EXPECT_EQ(*(mean.data), expected_data);
}

TEST(TensorTest, TestMeanAllReduceKeep){
    std::vector<float> expected_data = {12.5};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(INT_MIN, true);
    EXPECT_EQ(*(mean.data), expected_data);
}


TEST(TensorTest, TestMeanNDimReduceDontKeep){
    std::vector<float> expected_data = {5.0,6.0,7.0,8.0,17.0,18.0,19.0,20.0};
    std::vector<int> expect_shape = {2,4};
    std::vector<int> expect_strides = {4,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(1,false);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}

TEST(TensorTest, TestMeanNDimReduceKeep){
    std::vector<float> expected_data = {5.0,6.0,7.0,8.0,17.0,18.0,19.0,20.0};
    std::vector<int> expect_shape = {2,1,4};
    std::vector<int> expect_strides = {4,4,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(1,true);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}

TEST(TensorTest, TestMeanLastDimReduceDontKeep){
    std::vector<float> expected_data = {2.5,  6.5, 10.5, 14.5, 18.5, 22.5};
    std::vector<int> expect_shape = {2,3};
    std::vector<int> expect_strides = {3,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(-1,false);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}

TEST(TensorTest, TestMeanLastDimReduceKeep){
    std::vector<float> expected_data = {2.5,  6.5, 10.5, 14.5, 18.5, 22.5};
    std::vector<int> expect_shape = {2,3,1};
    std::vector<int> expect_strides = {3,1,1};
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(-1,true);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}

TEST(TensorTest, TestMeanOneDimReduceDontKeep){
    std::vector<float> expected_data = {5.5};
    std::vector<int> expect_shape = {1};
    std::vector<int> expect_strides = {1};
    std::vector<int> input_shape = {10};
    std::vector<float> input_data = {1,2,3,4,5,6,7,8,9,10};
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(0, false);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}

TEST(TensorTest, TestMeanOneDimReduceKeep){
    std::vector<float> expected_data = {5.5};
    std::vector<int> expect_shape = {1};
    std::vector<int> expect_strides = {1};
    std::vector<int> input_shape = {10};
    std::vector<float> input_data = {1,2,3,4,5,6,7,8,9,10};
    EzTensor::Tensor t(input_shape, input_data); 
    EzTensor::Tensor mean = t.mean(0, true);
    EXPECT_EQ(*(mean.data), expected_data);
    EXPECT_EQ(mean.shape, expect_shape);
    EXPECT_EQ(mean.strides,expect_strides);
}


TEST(TensorTest, TestMeanInvalidDim){
    std::vector<int> input_shape = {2,3,4};
    std::vector<float> input_data;
    for (float i = 1.0; i <= 24.0; ++i) {
        input_data.push_back(i);
    }
    EzTensor::Tensor t(input_shape, input_data); 
    EXPECT_THROW(
        {
        try {
            EzTensor::Tensor mean = t.mean(3,false);
        }
        catch(std::runtime_error& exception){
            EXPECT_STREQ("Invalid reduction dimension", exception.what() );
            throw;
        }
        }, std::runtime_error);
}

TEST(TensorTest, TestExpand){

    
}
