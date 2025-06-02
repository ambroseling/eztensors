add_test([=[AttentionTest.TestRMSNormLayer]=]  /Users/ambroseling/Projects/eztensors/build/test_rmsnorm [==[--gtest_filter=AttentionTest.TestRMSNormLayer]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[AttentionTest.TestRMSNormLayer]=]  PROPERTIES WORKING_DIRECTORY /Users/ambroseling/Projects/eztensors/build SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_rmsnorm_TESTS AttentionTest.TestRMSNormLayer)
