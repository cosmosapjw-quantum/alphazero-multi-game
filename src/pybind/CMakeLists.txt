if(ALPHAZERO_ENABLE_PYTHON AND pybind11_FOUND)
    # Create the Python module
    pybind11_add_module(alphazero_py python_bindings.cpp)
    
    target_include_directories(alphazero_py PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    
    # Link with available libraries - use if exists pattern to avoid errors
    # if libraries are not built yet
    if(TARGET alphazero_core)
        target_link_libraries(alphazero_py PRIVATE alphazero_core)
    endif()
    
    if(TARGET alphazero_mcts)
        target_link_libraries(alphazero_py PRIVATE alphazero_mcts)
    endif()
    
    if(TARGET alphazero_nn)
        target_link_libraries(alphazero_py PRIVATE alphazero_nn)
    endif()
    
    # Check for game implementations
    if(TARGET alphazero_gomoku)
        target_link_libraries(alphazero_py PRIVATE alphazero_gomoku)
    endif()
    
    if(TARGET alphazero_chess)
        target_link_libraries(alphazero_py PRIVATE alphazero_chess)
    endif()
    
    if(TARGET alphazero_go)
        target_link_libraries(alphazero_py PRIVATE alphazero_go)
    endif()
    
    # Additional components
    if(TARGET alphazero_ui)
        target_link_libraries(alphazero_py PRIVATE alphazero_ui)
    endif()
    
    if(TARGET alphazero_selfplay)
        target_link_libraries(alphazero_py PRIVATE alphazero_selfplay)
    endif()
    
    if(TARGET alphazero_elo)
        target_link_libraries(alphazero_py PRIVATE alphazero_elo)
    endif()
    
    # Always link with Torch libraries
    target_link_libraries(alphazero_py PRIVATE ${TORCH_LIBRARIES})
    
    set_target_properties(alphazero_py PROPERTIES
        OUTPUT_NAME "alphazero"
    )
    
    # Copy Python module to the project directory
    add_custom_command(TARGET alphazero_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:alphazero_py>
        ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:alphazero_py>
    )
else()
    message(STATUS "Skipping Python bindings (pybind11 not found or Python bindings disabled)")
endif() 