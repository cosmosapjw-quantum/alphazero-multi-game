if(ALPHAZERO_ENABLE_PYTHON AND pybind11_FOUND)
    # Create the Python module with static linking
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(alphazero_py SHARED python_bindings.cpp)
    
    target_include_directories(alphazero_py PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    
    # Find and explicitly link with the C10 library
    find_library(C10_LIBRARY c10 PATHS ${TORCH_INSTALL_PREFIX}/lib ${CMAKE_PREFIX_PATH}/lib)
    if(C10_LIBRARY)
        message(STATUS "Found c10 library: ${C10_LIBRARY}")
    else()
        message(WARNING "c10 library not found!")
    endif()
    
    # Link with all available libraries
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
    
    # Explicitly link with PyTorch libraries
    target_link_libraries(alphazero_py PRIVATE 
        ${TORCH_LIBRARIES}
        ${C10_LIBRARY}
    )
    
    # Explicitly link with fmt library to resolve symbol issues
    target_link_libraries(alphazero_py PRIVATE fmt::fmt)
    
    # Add other common dependencies
    target_link_libraries(alphazero_py PRIVATE 
        spdlog::spdlog 
        nlohmann_json::nlohmann_json
    )
    
    # Set RPATH to include the PyTorch library directory
    if(APPLE)
        set_target_properties(alphazero_py PROPERTIES
            INSTALL_RPATH "@loader_path;@loader_path/../lib;${TORCH_INSTALL_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(alphazero_py PROPERTIES
            INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${TORCH_INSTALL_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    
    set_target_properties(alphazero_py PROPERTIES
        OUTPUT_NAME "_alphazero_cpp"
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