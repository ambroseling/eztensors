cmake_minimum_required(VERSION 3.10)
project(EzTensors)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  test_tensor
  tests/test_tensor.cpp
)
add_executable(
  test_attention
  tests/test_attention.cpp
)
add_executable(
  test_rmsnorm
  tests/test_rmsnorm.cpp
)
target_link_libraries(
  test_tensor
  GTest::gtest_main
)
target_link_libraries(
  test_attention
  GTest::gtest_main
)
target_link_libraries(
  test_rmsnorm
  GTest::gtest_main
)


include(GoogleTest)

gtest_discover_tests(test_tensor)
gtest_discover_tests(test_attention)
gtest_discover_tests(test_rmsnorm)


add_executable(eztensors main.cpp)
