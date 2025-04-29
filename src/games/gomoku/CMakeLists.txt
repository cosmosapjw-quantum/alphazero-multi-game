add_library(alphazero_gomoku
    gomoku_state.cpp
    gomoku_rules.cpp
)

target_include_directories(alphazero_gomoku PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_gomoku PUBLIC
    alphazero_core
)