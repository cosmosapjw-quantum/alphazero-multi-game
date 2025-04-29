# CUDA test example
add_executable(cuda_test
    cuda_test.cpp
)

target_link_libraries(cuda_test
    alphazero_core
    ${TORCH_LIBRARIES}
)

# Add game examples
add_executable(chess_example
    chess_example.cpp
)

target_link_libraries(chess_example
    alphazero_core
    alphazero_chess
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)

add_executable(go_example
    go_example.cpp
)

target_link_libraries(go_example
    alphazero_core
    alphazero_go
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)

add_executable(gomoku_example
    gomoku_example.cpp
)

target_link_libraries(gomoku_example
    alphazero_core
    alphazero_gomoku
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)

# MCTS and NN integration test
add_executable(test_mcts_nn
    test_mcts_nn.cpp
)

target_link_libraries(test_mcts_nn
    alphazero_core
    alphazero_chess
    alphazero_go
    alphazero_gomoku
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)

# Install the examples
install(TARGETS 
    cuda_test
    chess_example
    go_example
    gomoku_example
    test_mcts_nn
    RUNTIME DESTINATION bin/examples
)