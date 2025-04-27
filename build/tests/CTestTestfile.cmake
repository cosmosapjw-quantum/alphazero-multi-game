# CMake generated Testfile for 
# Source directory: /home/cosmos/alphazero-multi-game/tests
# Build directory: /home/cosmos/alphazero-multi-game/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/cosmos/alphazero-multi-game/build/tests/core_tests[1]_include.cmake")
include("/home/cosmos/alphazero-multi-game/build/tests/game_tests[1]_include.cmake")
include("/home/cosmos/alphazero-multi-game/build/tests/mcts_tests[1]_include.cmake")
include("/home/cosmos/alphazero-multi-game/build/tests/nn_tests[1]_include.cmake")
include("/home/cosmos/alphazero-multi-game/build/tests/integration_tests[1]_include.cmake")
add_test(performance_tests "/home/cosmos/alphazero-multi-game/build/bin/performance_tests")
set_tests_properties(performance_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/cosmos/alphazero-multi-game/tests/CMakeLists.txt;88;add_test;/home/cosmos/alphazero-multi-game/tests/CMakeLists.txt;0;")
