add_library(alphazero_go
    go_state.cpp
    go_rules.cpp
)

target_include_directories(alphazero_go PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_go PUBLIC
    alphazero_core
)