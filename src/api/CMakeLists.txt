add_library(alphazero_api
    http_server.cpp
    rest_api.cpp
    server_main.cpp
)

target_include_directories(alphazero_api PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_api PUBLIC
    alphazero_core
    alphazero_mcts
    alphazero_nn
    ${TORCH_LIBRARIES}
)