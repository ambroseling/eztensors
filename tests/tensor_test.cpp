#include <gtest/gtest.h>
#include "../src/tensor.cpp"

TEST(TensorTest, BasicAssertions) {
std::vector<int> shape_1 = {2,2};
EzTensor::Tensor t1(shape_1);
t1.fill_with(5);
std::vector<int> check_shape = {2,2};
EXPECT_EQ(t1.shape, check_shape);
}