add_library(alphazero_mcts
    mcts_node.cpp
    parallel_mcts.cpp
    thread_pool.cpp
    transposition_table.cpp
)

target_include_directories(alphazero_mcts PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_mcts PUBLIC
    alphazero_core
    ${CMAKE_THREAD_LIBS_INIT}
    ${TORCH_LIBRARIES}
) 