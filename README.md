# EzTensors

EzTensors is a neural network library written in C / C++ from scratch. This project is not intended to be used for production but mainly for educational and research purposes. This project uses GTest as the testing suite and CMake as the build tool for the project, all CMake configurations can be found in CMakeLists.txt. The optimizations chosen are still limited at the moment since this is still a work in progress as I continously learn more and more about neural network libraries and modern AI engines.

## Implementation Roadmap
[] Llama 3.2 - (50% done)

## Some resources that are pretty cool:
- https://salykova.github.io/matmul-cpu
- https://github.com/meta-llama/llama
- https://andrewkchan.dev/posts/yalm.html
- https://mesozoic-egg.github.io/tinygrad-notes/


## Quick Overivew of Tensor Class
- Tensors are arranged in row-major format in memory, meaning accessing elements sequentially is equivalent to a index-by-index increment from smallest dimension to largest.
- Tensors have a `data` attribute that is encapsulated with a shared pointer. 
- Just like torch, `view` only works on contiguous tensors and points to the same memory as the original tensor just with different stride and shape metadata, `reshape` returns a view if the tensor is contiguous. If the tensor was not contiguous it performs a copy with contiguous layout then returns a view
- Matrix Muliplitcaions 
- Tensors have 4 main operations
    - 1. Tensor Manipulations
    - 2. Element-Wise Operations
    - 3. Reduction Operations
    - 4. Matrix Mulitplications and Convolutions
    - 5. (Bonus) Fused operations (TBD)
                                                      

## Installation
1. First clone the repository
```
git clone https://github.com/ambroseling/eztensors.git
```

2. Install CMake
```
brew install cmake
```

3. Build and compile the project
```cpp
// Generate build files in the build directory based on the CMakeLists.txt in the source directory in .
// CMake scans CMakeLists.txt to find the compiler, look for dependencies, generate build rules and configures builkd targets for executables
cmake -S . -B build
// Compiles the project using the generated build files
cmake --build build
```

4. Run Tests
```cpp
// This runs all the tests registed under `add_test()` in CMakeLists.txt
cd build && ctest
```

## Getting Started With EzTensor
```

```                                                                                                       

## Example Usage
```

```  

### Creating Tensors
```

```  

### Arithmetic Operations
```

```  

### Matrix Multiplication
```

```  

### Model Inference
```

```  

## Takeaways
