# Gomoku example
add_executable(gomoku_example
    gomoku_example.cpp
)
target_link_libraries(gomoku_example
    PRIVATE
    alphazero_lib
)

# Chess example (conditionally add if file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chess_example.cpp")
    add_executable(chess_example
        chess_example.cpp
    )
    target_link_libraries(chess_example
        PRIVATE
        alphazero_lib
    )
endif()

# Go example (conditionally add if file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/go_example.cpp")
    add_executable(go_example
        go_example.cpp
    )
    target_link_libraries(go_example
        PRIVATE
        alphazero_lib
    )
endif()

# MCTS Neural Network Test
add_executable(test_mcts_nn
    test_mcts_nn.cpp
)
target_link_libraries(test_mcts_nn
    PRIVATE
    alphazero_lib
)

# Install examples
install(TARGETS gomoku_example test_mcts_nn
    RUNTIME DESTINATION bin/examples
)

if(TARGET chess_example)
    install(TARGETS chess_example
        RUNTIME DESTINATION bin/examples
    )
endif()

if(TARGET go_example)
    install(TARGETS go_example
        RUNTIME DESTINATION bin/examples
    )
endif()