// include/alphazero/ui/game_ui.h
#ifndef GAME_UI_H
#define GAME_UI_H

#include <string>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include "alphazero/core/igamestate.h"
#include "alphazero/ui/renderer.h"

namespace alphazero {
namespace ui {

/**
 * @brief Game user interface for displaying and interacting with games
 * 
 * This class provides functionality for displaying game boards and
 * interacting with the game through a user interface.
 */
class GameUI {
public:
    /**
     * @brief Constructor
     * 
     * @param renderer Renderer for drawing the game
     */
    explicit GameUI(std::shared_ptr<Renderer> renderer = nullptr);
    
    /**
     * @brief Destructor
     */
    ~GameUI();
    
    /**
     * @brief Set the game state
     * 
     * @param state Game state to display
     */
    void setGameState(std::shared_ptr<core::IGameState> state);
    
    /**
     * @brief Get the game state
     * 
     * @return Current game state
     */
    std::shared_ptr<core::IGameState> getGameState() const { return state_; }
    
    /**
     * @brief Set the renderer
     * 
     * @param renderer Renderer to use
     */
    void setRenderer(std::shared_ptr<Renderer> renderer);
    
    /**
     * @brief Draw the current game state
     * 
     * @param highlightMove Move to highlight (if any)
     */
    void draw(int highlightMove = -1);
    
    /**
     * @brief Handle mouse click at position
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @return Action corresponding to the click, or -1 if no valid action
     */
    int handleClick(int x, int y);
    
    /**
     * @brief Convert screen coordinates to board coordinates
     * 
     * @param x X coordinate on screen
     * @param y Y coordinate on screen
     * @return Pair of board coordinates, or (-1, -1) if outside board
     */
    std::pair<int, int> screenToBoardCoordinates(int x, int y) const;
    
    /**
     * @brief Convert board coordinates to screen coordinates
     * 
     * @param row Board row
     * @param col Board column
     * @return Pair of screen coordinates
     */
    std::pair<int, int> boardToScreenCoordinates(int row, int col) const;
    
    /**
     * @brief Convert board coordinates to action
     * 
     * @param row Board row
     * @param col Board column
     * @return Action corresponding to the coordinates, or -1 if invalid
     */
    int boardCoordinatesToAction(int row, int col) const;
    
    /**
     * @brief Convert action to board coordinates
     * 
     * @param action Action to convert
     * @return Pair of board coordinates, or (-1, -1) if invalid action
     */
    std::pair<int, int> actionToBoardCoordinates(int action) const;
    
    /**
     * @brief Set the callback for move events
     * 
     * @param callback Function to call when a move is made
     */
    void setMoveCallback(std::function<void(int)> callback) {
        moveCallback_ = callback;
    }
    
    /**
     * @brief Set the board size in pixels
     * 
     * @param width Width in pixels
     * @param height Height in pixels
     */
    void setBoardSize(int width, int height);
    
    /**
     * @brief Get the board size in pixels
     * 
     * @return Pair of (width, height) in pixels
     */
    std::pair<int, int> getBoardSize() const;
    
    /**
     * @brief Set the cell size in pixels
     * 
     * @param cellSize Cell size in pixels
     */
    void setCellSize(int cellSize);
    
    /**
     * @brief Get the cell size in pixels
     * 
     * @return Cell size in pixels
     */
    int getCellSize() const { return cellSize_; }
    
    /**
     * @brief Highlight the last move
     * 
     * @param highlight Whether to highlight the last move
     */
    void setHighlightLastMove(bool highlight) { highlightLastMove_ = highlight; }
    
    /**
     * @brief Get whether the last move is highlighted
     * 
     * @return true if the last move is highlighted, false otherwise
     */
    bool getHighlightLastMove() const { return highlightLastMove_; }
    
    /**
     * @brief Show move numbers on the board
     * 
     * @param show Whether to show move numbers
     */
    void setShowMoveNumbers(bool show) { showMoveNumbers_ = show; }
    
    /**
     * @brief Get whether move numbers are shown
     * 
     * @return true if move numbers are shown, false otherwise
     */
    bool getShowMoveNumbers() const { return showMoveNumbers_; }
    
    /**
     * @brief Show coordinates on the board
     * 
     * @param show Whether to show coordinates
     */
    void setShowCoordinates(bool show) { showCoordinates_ = show; }
    
    /**
     * @brief Get whether coordinates are shown
     * 
     * @return true if coordinates are shown, false otherwise
     */
    bool getShowCoordinates() const { return showCoordinates_; }
    
    /**
     * @brief Render the game state to a string
     * 
     * @return String representation of the game state
     */
    std::string renderToString();
    
private:
    std::shared_ptr<core::IGameState> state_;      // Current game state
    std::shared_ptr<Renderer> renderer_;           // Renderer for drawing
    
    // Board dimensions
    int boardWidth_ = 600;                         // Board width in pixels
    int boardHeight_ = 600;                        // Board height in pixels
    int cellSize_ = 30;                            // Cell size in pixels
    int marginX_ = 30;                             // Horizontal margin in pixels
    int marginY_ = 30;                             // Vertical margin in pixels
    
    // Display options
    bool highlightLastMove_ = true;                // Whether to highlight the last move
    bool showMoveNumbers_ = false;                 // Whether to show move numbers
    bool showCoordinates_ = true;                  // Whether to show coordinates
    
    // Callback for move events
    std::function<void(int)> moveCallback_;
    
    // Game-specific rendering functions
    void drawGomoku(int highlightMove);
    void drawChess(int highlightMove);
    void drawGo(int highlightMove);
    
    // Game-specific coordinate conversion
    int gomokuBoardToAction(int row, int col) const;
    std::pair<int, int> gomokuActionToBoard(int action) const;
    int chessBoardToAction(int row, int col) const;
    std::pair<int, int> chessActionToBoard(int action) const;
    int goBoardToAction(int row, int col) const;
    std::pair<int, int> goActionToBoard(int action) const;
};

} // namespace ui
} // namespace alphazero

#endif // GAME_UI_H