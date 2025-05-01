add_library(alphazero_nn
    neural_network.cpp
    torch_neural_network.cpp
    batch_queue.cpp
    attack_defense_module.cpp
    random_policy_network.cpp
    ddw_randwire_resnet.cpp
)

target_include_directories(alphazero_nn PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_nn PUBLIC
    alphazero_core
    alphazero_mcts
    ${TORCH_LIBRARIES}
) 