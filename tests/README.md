# EzTensor Test Cases Documentation

For more information on GTest, please view https://google.github.io/googletest/reference/testing.html

## Constructor Tests
- **TestCreationDefinedShape**: Verifies tensor creation with specified dimensions
- **TestCreationZeroShape**: Ensures proper error handling when creating tensor with zero dimensions
- **TestCreationShared**: Tests creation of tensor that shares memory with another tensor
- **TestCreationInputData**: Validates tensor creation with predefined input data
- **TestCreationInputDataWrongShape**: Verifies error handling when input data doesn't match specified shape
- **TestFillWith**: Tests filling tensor with a constant value
- **TestFillWithScalar**: Validates filling a scalar (1D) tensor

## Arithmetic Operation Tests

### Addition Tests
- **TestAddWrongShape**: Verifies error handling for addition of incompatible tensor shapes
- **TestAddDiffTensor**: Tests addition of two different tensors
- **TestAddSameTensor**: Validates addition of tensor with itself
- **TestAddFloat**: Tests addition of tensor with scalar value
- **TestAddInplace**: Verifies in-place addition operation

### Subtraction Tests
- **TestSubtractWrongShape**: Ensures proper error handling for subtraction of incompatible shapes
- **TestSubtractDiffTensor**: Tests subtraction between different tensors
- **TestSubtractSameTensor**: Validates subtraction of tensor from itself
- **TestSubtractFloat**: Tests subtraction of scalar from tensor
- **TestSubtractInplace**: Verifies in-place subtraction operation

### Multiplication Tests
- **TestMultiplyWrongShape**: Validates error handling for multiplication of incompatible shapes
- **TestMultiplyDiffTensor**: Tests element-wise multiplication of different tensors
- **TestMultiplySameTensor**: Verifies multiplication of tensor by itself
- **TestMultiplyFloat**: Tests multiplication by scalar value
- **TestMultiplyInplace**: Validates in-place multiplication operation

### Division Tests
- **TestDivideWrongShape**: Ensures proper error handling for division with incompatible shapes
- **TestDivideDiffTensor**: Tests element-wise division of tensors
- **TestDivideSameTensor**: Validates division of tensor by itself
- **TestDivideFloat**: Tests division by scalar value
- **TestDivideInplace**: Verifies in-place division operation

## Reshape and View Tests
- **TestView**: Validates reshaping tensor while maintaining data continuity
- **TestViewNonContig**: Ensures error handling for viewing non-contiguous tensor
- **TestTransposeContig**: Tests transposition and contiguous conversion

## Reduction Operation Tests

### Sum Operation Tests
- **TestSumAllReduceDontKeep**: Tests reduction across all dimensions without keeping dims
- **TestSumAllReduceKeep**: Validates reduction across all dimensions keeping dims
- **TestSumNDimReduceDontKeep**: Tests reduction along specified dimension without keeping dims
- **TestSumNDimReduceKeep**: Validates reduction along specified dimension keeping dims
- **TestSumLastDimReduceDontKeep**: Tests reduction along last dimension without keeping dims
- **TestSumLastDimReduceKeep**: Validates reduction along last dimension keeping dims
- **TestSumOneDimReduceDontKeep**: Tests reduction of 1D tensor without keeping dims
- **TestSumOneDimReduceKeep**: Validates reduction of 1D tensor keeping dims
- **TestSumInvalidDim**: Ensures proper error handling for invalid reduction dimension

### Mean Operation Tests
- **TestMeanAllReduceDontKeep**: Tests mean across all dimensions without keeping dims
- **TestMeanAllReduceKeep**: Validates mean across all dimensions keeping dims
- **TestMeanNDimReduceDontKeep**: Tests mean along specified dimension without keeping dims
- **TestMeanNDimReduceKeep**: Validates mean along specified dimension keeping dims
- **TestMeanLastDimReduceDontKeep**: Tests mean along last dimension without keeping dims
- **TestMeanLastDimReduceKeep**: Validates mean along last dimension keeping dims
- **TestMeanOneDimReduceDontKeep**: Tests mean of 1D tensor without keeping dims
- **TestMeanOneDimReduceKeep**: Validates mean of 1D tensor keeping dims
- **TestMeanInvalidDim**: Ensures proper error handling for invalid reduction dimension

## Expand Operation
- **TestExpand**: (Test implementation pending) Will test tensor expansion functionality

Each test case uses Google Test framework and follows the Arrange-Act-Assert pattern to validate the functionality of the EzTensor library.