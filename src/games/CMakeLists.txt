add_subdirectory(chess)
add_subdirectory(go)
add_subdirectory(gomoku) 

# Create a combined games library
add_library(alphazero_games INTERFACE)

target_link_libraries(alphazero_games INTERFACE
    alphazero_chess
    alphazero_go
    alphazero_gomoku
)

# Install the library
install(TARGETS alphazero_games
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
) 