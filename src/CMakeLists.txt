add_subdirectory(core)
add_subdirectory(mcts)
add_subdirectory(nn)
add_subdirectory(games)
add_subdirectory(selfplay)
add_subdirectory(elo)

if(ALPHAZERO_BUILD_GUI)
    add_subdirectory(ui)
endif()

if(ALPHAZERO_ENABLE_PYTHON)
    add_subdirectory(pybind)
endif()

if(ALPHAZERO_BUILD_API)
    add_subdirectory(api)
endif()

if(ALPHAZERO_BUILD_CLI)
    add_subdirectory(cli)
endif()

# Add the main AlphaZero libraries
add_library(alphazero_lib STATIC
    core/AlphaZero.cpp
    core/game_factory.cpp
    core/igamestate.cpp
    core/zobrist_hash.cpp
    mcts/mcts_node.cpp
    mcts/parallel_mcts.cpp
    mcts/thread_pool.cpp
    mcts/transposition_table.cpp
    nn/attack_defense_module.cpp
    nn/batch_queue.cpp
    nn/neural_network.cpp
    nn/random_policy_network.cpp
    nn/torch_neural_network.cpp
    selfplay/dataset.cpp
    selfplay/game_record.cpp
    selfplay/self_play_manager.cpp
    games/chess/chess_state.cpp
    games/chess/chess_rules.cpp
    games/chess/chess960.cpp
    games/go/go_state.cpp
    games/go/go_rules.cpp
    games/gomoku/gomoku_state.cpp
    games/gomoku/gomoku_rules.cpp
)

target_include_directories(alphazero_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_lib PUBLIC
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Install the library
install(TARGETS alphazero_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Also keep the interface library for compatibility
add_library(alphazero INTERFACE)
target_include_directories(alphazero INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(alphazero INTERFACE alphazero_lib)