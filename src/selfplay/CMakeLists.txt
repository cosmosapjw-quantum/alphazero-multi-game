add_library(alphazero_selfplay
    dataset.cpp
    game_record.cpp
    self_play_manager.cpp
)

target_include_directories(alphazero_selfplay PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_selfplay PUBLIC
    alphazero_core
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)