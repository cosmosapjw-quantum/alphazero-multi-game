add_library(alphazero_chess
    chess_state.cpp
    chess_rules.cpp
    chess960.cpp
)

target_include_directories(alphazero_chess PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_chess PUBLIC
    alphazero_core
) 