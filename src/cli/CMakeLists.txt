add_library(alphazero_cli
    cli_interface.cpp
    cli_main.cpp
    command_parser.cpp
)

target_include_directories(alphazero_cli PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_cli PUBLIC
    alphazero_core
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)