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

# Add standalone self-play executable
add_executable(self_play
    selfplay_main.cpp
)

target_include_directories(self_play PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(self_play PUBLIC
    alphazero_selfplay
    alphazero_core
    alphazero_mcts
    alphazero_nn
    alphazero_games
    ${TORCH_LIBRARIES}
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Install the self-play executable
install(TARGETS self_play
    RUNTIME DESTINATION bin
)