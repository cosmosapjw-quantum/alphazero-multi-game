add_library(alphazero_ui
    game_ui.cpp
    renderer.cpp
)

target_include_directories(alphazero_ui PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alphazero_ui PUBLIC
    alphazero_core
)