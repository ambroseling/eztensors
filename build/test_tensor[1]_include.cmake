if(EXISTS "/Users/aling/Projects/eztensors/build/test_tensor[1]_tests.cmake")
  include("/Users/aling/Projects/eztensors/build/test_tensor[1]_tests.cmake")
else()
  add_test(test_tensor_NOT_BUILT test_tensor_NOT_BUILT)
endif()
