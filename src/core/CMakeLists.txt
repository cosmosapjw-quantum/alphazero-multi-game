add_library(alphazero_core
    AlphaZero.cpp
    igamestate.cpp
    game_factory.cpp
    zobrist_hash.cpp
)

target_include_directories(alphazero_core PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(alphazero_core
    ${TORCH_LIBRARIES}
)

# Forward declare that games will be linked later
set_target_properties(alphazero_core PROPERTIES
    INTERFACE_LINK_LIBRARIES "alphazero_games"
)

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    target_include_directories(alphazero_core PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(alphazero_core ${CUDNN_LIBRARY})
endif() 