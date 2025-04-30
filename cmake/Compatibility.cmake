# Compatibility.cmake - Fixes for library compatibility in the AlphaZero project

# Make sure Torch targets exist
if(NOT TARGET Torch::Torch AND TARGET torch)
    add_library(Torch::Torch ALIAS torch)
endif()

# Make sure fmt target exists
# ---- fmt ------------------------------------------------------
find_package(fmt QUIET)
if(NOT TARGET fmt::fmt)
    # Fallback: header-only interface target
    find_path(FMT_INCLUDE_DIR fmt/core.h REQUIRED)
    add_library(fmt INTERFACE IMPORTED)
    target_include_directories(fmt INTERFACE ${FMT_INCLUDE_DIR})
    add_library(fmt::fmt ALIAS fmt)
endif()


# Make sure spdlog target exists
find_package(spdlog REQUIRED)

# Make sure nlohmann_json target exists
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG QUIET)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
        add_library(nlohmann_json INTERFACE IMPORTED)
        target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    endif()
endif()

# Additional compatibility fixes can be added here