if(EXISTS "/home/cosmos/alphazero-multi-game/build/tests/nn_tests[1]_tests.cmake")
  include("/home/cosmos/alphazero-multi-game/build/tests/nn_tests[1]_tests.cmake")
else()
  add_test(nn_tests_NOT_BUILT nn_tests_NOT_BUILT)
endif()
