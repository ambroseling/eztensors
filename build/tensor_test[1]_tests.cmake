add_test([=[TensorTest.BasicAssertions]=]  /Users/ambroseling/Projects/eztensors/build/tensor_test [==[--gtest_filter=TensorTest.BasicAssertions]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorTest.BasicAssertions]=]  PROPERTIES WORKING_DIRECTORY /Users/ambroseling/Projects/eztensors/build SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  tensor_test_TESTS TensorTest.BasicAssertions)
