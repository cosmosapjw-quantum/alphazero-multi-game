add_library(alphazero_elo
    elo_tracker.cpp
)

target_include_directories(alphazero_elo PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_elo PUBLIC
    alphazero_core
)