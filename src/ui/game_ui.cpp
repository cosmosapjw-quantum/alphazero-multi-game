// src/ui/game_ui.cpp
#include "alphazero/ui/game_ui.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace alphazero {
namespace ui {

GameUI::GameUI(std::shared_ptr<Renderer> renderer)
    : renderer_(renderer) {
    
    if (!renderer_) {
        renderer_ = Renderer::createTextRenderer();
    }
}

GameUI::~GameUI() {
    // Clean up resources
}

void GameUI::setGameState(std::shared_ptr<core::IGameState> state) {
    state_ = state;
    
    // Adjust board dimensions based on game type and size
    if (state_) {
        int boardSize = state_->getBoardSize();
        cellSize_ = 30;  // Default cell size
        
        switch (state_->getGameType()) {
            case core::GameType::GOMOKU:
            case core::GameType::GO:
                // Square board
                boardWidth_ = boardHeight_ = cellSize_ * (boardSize + 2);  // +2 for margins
                marginX_ = marginY_ = cellSize_;
                break;
                
            case core::GameType::CHESS:
                // Square board with 8x8 grid
                boardWidth_ = boardHeight_ = cellSize_ * 10;  // +2 for margins
                marginX_ = marginY_ = cellSize_;
                break;
                
            default:
                // Default square board
                boardWidth_ = boardHeight_ = cellSize_ * (boardSize + 2);
                marginX_ = marginY_ = cellSize_;
                break;
        }
        
        if (renderer_) {
            renderer_->setSize(boardWidth_, boardHeight_);
        }
    }
}

void GameUI::setRenderer(std::shared_ptr<Renderer> renderer) {
    renderer_ = renderer;
    
    if (renderer_ && state_) {
        renderer_->setSize(boardWidth_, boardHeight_);
    }
}

void GameUI::draw(int highlightMove) {
    if (!state_ || !renderer_) {
        return;
    }
    
    // Clear the renderer
    renderer_->clear();
    
    // Draw based on game type
    switch (state_->getGameType()) {
        case core::GameType::GOMOKU:
            drawGomoku(highlightMove);
            break;
            
        case core::GameType::CHESS:
            drawChess(highlightMove);
            break;
            
        case core::GameType::GO:
            drawGo(highlightMove);
            break;
            
        default:
            // Default drawing (empty board)
            renderer_->drawRect(0, 0, boardWidth_, boardHeight_, "", "black", 1);
            break;
    }
    
    // Render the output
    renderer_->render();
}

int GameUI::handleClick(int x, int y) {
    if (!state_) {
        return -1;
    }
    
    // Convert screen coordinates to board coordinates
    auto [row, col] = screenToBoardCoordinates(x, y);
    
    if (row < 0 || col < 0) {
        return -1;  // Invalid coordinates
    }
    
    // Convert board coordinates to action
    int action = boardCoordinatesToAction(row, col);
    
    if (action >= 0 && state_->isLegalMove(action)) {
        // Valid action
        if (moveCallback_) {
            moveCallback_(action);
        }
        return action;
    }
    
    return -1;  // Invalid action
}

std::pair<int, int> GameUI::screenToBoardCoordinates(int x, int y) const {
    if (!state_) {
        return {-1, -1};
    }
    
    int boardSize = state_->getBoardSize();
    
    // Calculate grid position
    int col = (x - marginX_) / cellSize_;
    int row = (y - marginY_) / cellSize_;
    
    // Check if within board bounds
    if (col < 0 || col >= boardSize || row < 0 || row >= boardSize) {
        return {-1, -1};
    }
    
    return {row, col};
}

std::pair<int, int> GameUI::boardToScreenCoordinates(int row, int col) const {
    int x = marginX_ + col * cellSize_ + cellSize_ / 2;
    int y = marginY_ + row * cellSize_ + cellSize_ / 2;
    return {x, y};
}

int GameUI::boardCoordinatesToAction(int row, int col) const {
    if (!state_) {
        return -1;
    }
    
    switch (state_->getGameType()) {
        case core::GameType::GOMOKU:
            return gomokuBoardToAction(row, col);
            
        case core::GameType::CHESS:
            return chessBoardToAction(row, col);
            
        case core::GameType::GO:
            return goBoardToAction(row, col);
            
        default:
            return -1;
    }
}

std::pair<int, int> GameUI::actionToBoardCoordinates(int action) const {
    if (!state_) {
        return {-1, -1};
    }
    
    switch (state_->getGameType()) {
        case core::GameType::GOMOKU:
            return gomokuActionToBoard(action);
            
        case core::GameType::CHESS:
            return chessActionToBoard(action);
            
        case core::GameType::GO:
            return goActionToBoard(action);
            
        default:
            return {-1, -1};
    }
}

void GameUI::setBoardSize(int width, int height) {
    boardWidth_ = width;
    boardHeight_ = height;
    
    if (renderer_) {
        renderer_->setSize(width, height);
    }
}

std::pair<int, int> GameUI::getBoardSize() const {
    return {boardWidth_, boardHeight_};
}

void GameUI::setCellSize(int cellSize) {
    cellSize_ = cellSize;
    
    // Update board dimensions
    if (state_) {
        int boardSize = state_->getBoardSize();
        
        switch (state_->getGameType()) {
            case core::GameType::GOMOKU:
            case core::GameType::GO:
                // Square board
                boardWidth_ = boardHeight_ = cellSize_ * (boardSize + 2);
                marginX_ = marginY_ = cellSize_;
                break;
                
            case core::GameType::CHESS:
                // Square board with 8x8 grid
                boardWidth_ = boardHeight_ = cellSize_ * 10;
                marginX_ = marginY_ = cellSize_;
                break;
                
            default:
                // Default square board
                boardWidth_ = boardHeight_ = cellSize_ * (boardSize + 2);
                marginX_ = marginY_ = cellSize_;
                break;
        }
        
        if (renderer_) {
            renderer_->setSize(boardWidth_, boardHeight_);
        }
    }
}

std::string GameUI::renderToString() {
    if (!state_) {
        return "";
    }
    
    // Create a text renderer if we don't have one
    auto textRenderer = std::dynamic_pointer_cast<TextRenderer>(renderer_);
    if (!textRenderer) {
        textRenderer = std::dynamic_pointer_cast<TextRenderer>(Renderer::createTextRenderer());
        
        if (!textRenderer) {
            return state_->toString();  // Fallback to state's toString
        }
        
        // Temporarily use the text renderer
        auto oldRenderer = renderer_;
        renderer_ = textRenderer;
        draw();
        renderer_ = oldRenderer;
    } else {
        // Use existing text renderer
        draw();
    }
    
    return textRenderer->getText();
}

// Game-specific implementations

void GameUI::drawGomoku(int highlightMove) {
    if (!state_ || !renderer_) {
        return;
    }
    
    int boardSize = state_->getBoardSize();
    
    // Draw the board background
    renderer_->drawRect(0, 0, boardWidth_, boardHeight_, "#f0d9b5", "", 0);
    
    // Draw grid lines
    for (int i = 0; i <= boardSize - 1; ++i) {
        // Horizontal lines
        renderer_->drawLine(
            marginX_, marginY_ + i * cellSize_,
            marginX_ + (boardSize - 1) * cellSize_, marginY_ + i * cellSize_,
            "black", 1
        );
        
        // Vertical lines
        renderer_->drawLine(
            marginX_ + i * cellSize_, marginY_,
            marginX_ + i * cellSize_, marginY_ + (boardSize - 1) * cellSize_,
            "black", 1
        );
    }
    
    // Draw star points (for larger boards)
    if (boardSize >= 13) {
        int starPoints[][2] = {
            {3, 3}, {3, boardSize / 2}, {3, boardSize - 4},
            {boardSize / 2, 3}, {boardSize / 2, boardSize / 2}, {boardSize / 2, boardSize - 4},
            {boardSize - 4, 3}, {boardSize - 4, boardSize / 2}, {boardSize - 4, boardSize - 4}
        };
        
        for (int i = 0; i < 9; ++i) {
            int x = marginX_ + starPoints[i][1] * cellSize_;
            int y = marginY_ + starPoints[i][0] * cellSize_;
            renderer_->drawCircle(x, y, 3, "black", "", 0);
        }
    }
    
    // Draw coordinates
    if (showCoordinates_) {
        for (int i = 0; i < boardSize; ++i) {
            // Column labels (letters)
            char colLabel = static_cast<char>('A' + (i >= 8 ? i + 1 : i));  // Skip 'I'
            renderer_->drawText(
                marginX_ + i * cellSize_, marginY_ - 15,
                std::string(1, colLabel), 12, "black", true
            );
            
            // Row labels (numbers)
            renderer_->drawText(
                marginX_ - 15, marginY_ + i * cellSize_,
                std::to_string(boardSize - i), 12, "black", true
            );
        }
    }
    
    // Get move history
    std::vector<int> moveHistory = state_->getMoveHistory();
    
    // Get last move for highlighting
    int lastMove = -1;
    if (!moveHistory.empty() && highlightLastMove_) {
        lastMove = moveHistory.back();
    }
    
    // Draw stones
    for (int i = 0; i < boardSize; ++i) {
        for (int j = 0; j < boardSize; ++j) {
            int action = gomokuBoardToAction(i, j);
            
            // Try to get stone color from state
            std::unique_ptr<core::IGameState> tempState = state_->clone();
            for (size_t m = 0; m < moveHistory.size(); ++m) {
                if (moveHistory[m] == action) {
                    int player = m % 2 == 0 ? 1 : 2;  // Player 1 starts
                    
                    int x = marginX_ + j * cellSize_;
                    int y = marginY_ + i * cellSize_;
                    
                    // Draw stone
                    std::string stoneColor = (player == 1) ? "black" : "white";
                    std::string outlineColor = (player == 1) ? "" : "black";
                    renderer_->drawCircle(x, y, cellSize_ / 2 - 2, stoneColor, outlineColor, 1);
                    
                    // Highlight last move
                    if (action == lastMove || action == highlightMove) {
                        renderer_->drawCircle(x, y, cellSize_ / 4, 
                                            (player == 1) ? "white" : "black", "", 0);
                    }
                    
                    // Show move number
                    if (showMoveNumbers_) {
                        renderer_->drawText(x, y, std::to_string(m + 1), cellSize_ / 3, 
                                          (player == 1) ? "white" : "black", true);
                    }
                    
                    break;
                }
            }
        }
    }
}

void GameUI::drawChess(int highlightMove) {
    if (!state_ || !renderer_) {
        return;
    }
    
    // Chess has a fixed 8x8 board
    int boardSize = 8;
    
    // Draw checkered board
    for (int i = 0; i < boardSize; ++i) {
        for (int j = 0; j < boardSize; ++j) {
            bool isLightSquare = (i + j) % 2 == 0;
            std::string squareColor = isLightSquare ? "#f0d9b5" : "#b58863";
            
            int x = marginX_ + j * cellSize_;
            int y = marginY_ + i * cellSize_;
            
            renderer_->drawRect(x, y, cellSize_, cellSize_, squareColor, "", 0);
            
            // Highlight square if it's the last move
            std::vector<int> moveHistory = state_->getMoveHistory();
            if (!moveHistory.empty() && highlightLastMove_) {
                int lastMove = moveHistory.back();
                auto [lastRow, lastCol] = chessActionToBoard(lastMove);
                
                if (i == lastRow && j == lastCol) {
                    renderer_->drawRect(x, y, cellSize_, cellSize_, "", "#ff0000", 2);
                }
            }
            
            // Highlight specified move
            if (highlightMove >= 0) {
                auto [hlRow, hlCol] = chessActionToBoard(highlightMove);
                if (i == hlRow && j == hlCol) {
                    renderer_->drawRect(x, y, cellSize_, cellSize_, "", "#00ff00", 2);
                }
            }
        }
    }
    
    // Draw coordinates
    if (showCoordinates_) {
        for (int i = 0; i < boardSize; ++i) {
            // Column labels (a-h)
            char colLabel = static_cast<char>('a' + i);
            renderer_->drawText(
                marginX_ + i * cellSize_ + cellSize_ / 2, marginY_ + boardSize * cellSize_ + 15,
                std::string(1, colLabel), 12, "black", true
            );
            
            // Row labels (1-8)
            renderer_->drawText(
                marginX_ - 15, marginY_ + i * cellSize_ + cellSize_ / 2,
                std::to_string(boardSize - i), 12, "black", true
            );
        }
    }
    
    // For a proper chess board rendering, we'd need to draw pieces based on FEN state
    // This is a simplified version that just uses text for pieces
    std::string fen = state_->toString();
    std::vector<std::string> fenRows;
    std::istringstream iss(fen);
    std::string fenRow;
    
    // Get board rows from FEN
    while (std::getline(iss, fenRow, '/')) {
        if (fenRow.find(' ') != std::string::npos) {
            fenRow = fenRow.substr(0, fenRow.find(' '));
        }
        fenRows.push_back(fenRow);
    }
    
    // Correct the number of FEN rows to 8
    if (fenRows.size() > 8) {
        fenRows.resize(8);
    }
    
    // Draw pieces
    for (int i = 0; i < boardSize && i < static_cast<int>(fenRows.size()); ++i) {
        int col = 0;
        
        for (char c : fenRows[i]) {
            if (col >= boardSize) break;
            
            if (c >= '1' && c <= '8') {
                col += c - '0';
            } else {
                int x = marginX_ + col * cellSize_ + cellSize_ / 2;
                int y = marginY_ + i * cellSize_ + cellSize_ / 2;
                
                std::string pieceSymbol;
                switch (c) {
                    case 'P': pieceSymbol = "♙"; break;
                    case 'N': pieceSymbol = "♘"; break;
                    case 'B': pieceSymbol = "♗"; break;
                    case 'R': pieceSymbol = "♖"; break;
                    case 'Q': pieceSymbol = "♕"; break;
                    case 'K': pieceSymbol = "♔"; break;
                    case 'p': pieceSymbol = "♟"; break;
                    case 'n': pieceSymbol = "♞"; break;
                    case 'b': pieceSymbol = "♝"; break;
                    case 'r': pieceSymbol = "♜"; break;
                    case 'q': pieceSymbol = "♛"; break;
                    case 'k': pieceSymbol = "♚"; break;
                    default: pieceSymbol = c;
                }
                
                bool isWhite = (c >= 'A' && c <= 'Z');
                std::string textColor = isWhite ? "white" : "black";
                std::string outlineColor = isWhite ? "black" : "";
                
                renderer_->drawText(x, y, pieceSymbol, cellSize_ * 2/3, textColor, true);
                col++;
            }
        }
    }
}

void GameUI::drawGo(int highlightMove) {
    if (!state_ || !renderer_) {
        return;
    }
    
    int boardSize = state_->getBoardSize();
    
    // Draw the board background
    renderer_->drawRect(0, 0, boardWidth_, boardHeight_, "#E7C091", "", 0);
    
    // Draw grid lines
    for (int i = 0; i < boardSize; ++i) {
        // Horizontal lines
        renderer_->drawLine(
            marginX_, marginY_ + i * cellSize_,
            marginX_ + (boardSize - 1) * cellSize_, marginY_ + i * cellSize_,
            "black", 1
        );
        
        // Vertical lines
        renderer_->drawLine(
            marginX_ + i * cellSize_, marginY_,
            marginX_ + i * cellSize_, marginY_ + (boardSize - 1) * cellSize_,
            "black", 1
        );
    }
    
    // Draw star points
    const int starPointRadius = 3;
    
    // Different star point positions based on board size
    std::vector<std::pair<int, int>> starPoints;
    
    if (boardSize == 19) {
        // 19x19 board: 9 star points
        starPoints = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
    } else if (boardSize == 13) {
        // 13x13 board: 5 star points
        starPoints = {
            {3, 3}, {3, 9},
            {6, 6},
            {9, 3}, {9, 9}
        };
    } else if (boardSize == 9) {
        // 9x9 board: 5 star points
        starPoints = {
            {2, 2}, {2, 6},
            {4, 4},
            {6, 2}, {6, 6}
        };
    }
    
    for (const auto& point : starPoints) {
        int row = point.first;
        int col = point.second;
        
        int x = marginX_ + col * cellSize_;
        int y = marginY_ + row * cellSize_;
        
        renderer_->drawCircle(x, y, starPointRadius, "black", "", 0);
    }
    
    // Draw coordinates
    if (showCoordinates_) {
        for (int i = 0; i < boardSize; ++i) {
            // Column labels (A-T, skipping I)
            char colLabel = static_cast<char>('A' + (i >= 8 ? i + 1 : i));
            renderer_->drawText(
                marginX_ + i * cellSize_, marginY_ - 15,
                std::string(1, colLabel), 12, "black", true
            );
            
            // Row labels (1-19)
            renderer_->drawText(
                marginX_ - 15, marginY_ + i * cellSize_,
                std::to_string(boardSize - i), 12, "black", true
            );
        }
    }
    
    // Get move history
    std::vector<int> moveHistory = state_->getMoveHistory();
    
    // Get last move for highlighting
    int lastMove = -1;
    if (!moveHistory.empty() && highlightLastMove_) {
        lastMove = moveHistory.back();
    }
    
    // Draw stones
    for (int i = 0; i < boardSize; ++i) {
        for (int j = 0; j < boardSize; ++j) {
            int action = goBoardToAction(i, j);
            
            // Try to get stone color from state
            std::unique_ptr<core::IGameState> tempState = state_->clone();
            for (size_t m = 0; m < moveHistory.size(); ++m) {
                if (moveHistory[m] == action) {
                    int player = m % 2 == 0 ? 1 : 2;  // Player 1 starts
                    
                    int x = marginX_ + j * cellSize_;
                    int y = marginY_ + i * cellSize_;
                    
                    // Draw stone
                    std::string stoneColor = (player == 1) ? "black" : "white";
                    std::string outlineColor = (player == 1) ? "" : "black";
                    renderer_->drawCircle(x, y, cellSize_ / 2 - 2, stoneColor, outlineColor, 1);
                    
                    // Highlight last move
                    if (action == lastMove || action == highlightMove) {
                        renderer_->drawCircle(x, y, cellSize_ / 4, 
                                            (player == 1) ? "white" : "black", "", 0);
                    }
                    
                    // Show move number
                    if (showMoveNumbers_) {
                        renderer_->drawText(x, y, std::to_string(m + 1), cellSize_ / 3, 
                                          (player == 1) ? "white" : "black", true);
                    }
                    
                    break;
                }
            }
        }
    }
}

// Game-specific coordinate conversions

int GameUI::gomokuBoardToAction(int row, int col) const {
    if (!state_) {
        return -1;
    }
    
    int boardSize = state_->getBoardSize();
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
        return -1;
    }
    
    return row * boardSize + col;
}

std::pair<int, int> GameUI::gomokuActionToBoard(int action) const {
    if (!state_) {
        return {-1, -1};
    }
    
    int boardSize = state_->getBoardSize();
    if (action < 0 || action >= boardSize * boardSize) {
        return {-1, -1};
    }
    
    int row = action / boardSize;
    int col = action % boardSize;
    
    return {row, col};
}

int GameUI::chessBoardToAction(int row, int col) const {
    if (!state_) {
        return -1;
    }
    
    // Chess moves are encoded differently - this is a simplified version
    // that just returns the square index. In a real implementation, we would
    // need to handle piece selection and legal moves properly.
    
    int boardSize = 8;  // Chess is 8x8
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
        return -1;
    }
    
    return row * boardSize + col;
}

std::pair<int, int> GameUI::chessActionToBoard(int action) const {
    if (!state_) {
        return {-1, -1};
    }
    
    // This is a simplified version that just decodes the square index
    int boardSize = 8;  // Chess is 8x8
    
    // Check if it's a valid square
    if (action < 0 || action >= boardSize * boardSize) {
        return {-1, -1};
    }
    
    int row = action / boardSize;
    int col = action % boardSize;
    
    return {row, col};
}

int GameUI::goBoardToAction(int row, int col) const {
    if (!state_) {
        return -1;
    }
    
    int boardSize = state_->getBoardSize();
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
        return -1;
    }
    
    return row * boardSize + col;
}

std::pair<int, int> GameUI::goActionToBoard(int action) const {
    if (!state_) {
        return {-1, -1};
    }
    
    int boardSize = state_->getBoardSize();
    
    // Check for pass move
    if (action == boardSize * boardSize) {
        return {-1, -1};  // Pass move
    }
    
    // Check if it's a valid position
    if (action < 0 || action >= boardSize * boardSize) {
        return {-1, -1};
    }
    
    int row = action / boardSize;
    int col = action % boardSize;
    
    return {row, col};
}

} // namespace ui
} // namespace alphazero