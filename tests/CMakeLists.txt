# Core tests
file(GLOB_RECURSE CORE_TEST_SOURCES "core/*.cpp")
if(CORE_TEST_SOURCES)
    add_executable(core_tests ${CORE_TEST_SOURCES})
    target_link_libraries(core_tests
        PRIVATE 
        alphazero_lib
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    gtest_discover_tests(core_tests)
endif()

# Game tests
file(GLOB_RECURSE GAME_TEST_SOURCES "games/*.cpp")
if(GAME_TEST_SOURCES)
    add_executable(game_tests ${GAME_TEST_SOURCES})
    target_link_libraries(game_tests
        PRIVATE 
        alphazero_lib
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    gtest_discover_tests(game_tests)
endif()

# MCTS tests
file(GLOB_RECURSE MCTS_TEST_SOURCES "mcts/*.cpp")
if(MCTS_TEST_SOURCES)
    add_executable(mcts_tests ${MCTS_TEST_SOURCES})
    target_link_libraries(mcts_tests
        PRIVATE 
        alphazero_lib
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    gtest_discover_tests(mcts_tests)
endif()

# Neural Network tests
file(GLOB_RECURSE NN_TEST_SOURCES "nn/*.cpp")
if(NN_TEST_SOURCES)
    add_executable(nn_tests ${NN_TEST_SOURCES})
    target_link_libraries(nn_tests
        PRIVATE 
        alphazero_lib
        alphazero_nn
        alphazero_games
        alphazero_chess
        alphazero_go
        alphazero_gomoku
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    gtest_discover_tests(nn_tests)
endif()

# Integration tests
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "integration/*.cpp")
if(INTEGRATION_TEST_SOURCES)
    add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
    target_link_libraries(integration_tests
        PRIVATE 
        alphazero_lib
        alphazero_nn
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    gtest_discover_tests(integration_tests)
endif()

# Performance tests
file(GLOB_RECURSE PERFORMANCE_TEST_SOURCES "performance/*.cpp")
if(PERFORMANCE_TEST_SOURCES)
    add_executable(performance_tests ${PERFORMANCE_TEST_SOURCES})
    target_link_libraries(performance_tests
        PRIVATE 
        alphazero_lib
        GTest::gtest
        GTest::gtest_main
        pthread
    )
    # Performance tests might be slow, so we don't auto-discover them
    add_test(NAME performance_tests COMMAND performance_tests)
endif()