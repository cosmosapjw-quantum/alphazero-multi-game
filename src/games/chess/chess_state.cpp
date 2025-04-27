// src/games/chess/chess_state.cpp
#include "alphazero/games/chess/chess_state.h"
#include <sstream>
#include <algorithm>
#include <map>
#include <cctype>
#include <stdexcept>
#include <iostream>
#include <array>

namespace alphazero {
namespace chess {

// Constructor implementations
ChessState::ChessState(bool chess960, const std::string& fen)
    : IGameState(core::GameType::CHESS),
      current_player_(PieceColor::WHITE),
      en_passant_square_(-1),
      halfmove_clock_(0),
      fullmove_number_(1),
      chess960_(chess960),
      legal_moves_dirty_(true),
      hash_dirty_(true),
      terminal_check_dirty_(true),
      zobrist_(core::GameType::CHESS, 8, 12) {  // 6 piece types * 2 colors
    
    if (!fen.empty()) {
        if (!setFromFEN(fen)) {
            // If FEN parsing fails, use starting position
            initializeStartingPosition();
        }
    } else {
        initializeStartingPosition();
    }
}

// Copy constructor
ChessState::ChessState(const ChessState& other)
    : IGameState(core::GameType::CHESS),
      board_(other.board_),
      current_player_(other.current_player_),
      castling_rights_(other.castling_rights_),
      en_passant_square_(other.en_passant_square_),
      halfmove_clock_(other.halfmove_clock_),
      fullmove_number_(other.fullmove_number_),
      chess960_(other.chess960_),
      move_history_(other.move_history_),
      legal_moves_dirty_(true),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_),
      is_terminal_cached_(other.is_terminal_cached_),
      cached_result_(other.cached_result_),
      terminal_check_dirty_(other.terminal_check_dirty_),
      zobrist_(core::GameType::CHESS, 8, 12) {
}

// Assignment operator
ChessState& ChessState::operator=(const ChessState& other) {
    if (this != &other) {
        board_ = other.board_;
        current_player_ = other.current_player_;
        castling_rights_ = other.castling_rights_;
        en_passant_square_ = other.en_passant_square_;
        halfmove_clock_ = other.halfmove_clock_;
        fullmove_number_ = other.fullmove_number_;
        chess960_ = other.chess960_;
        move_history_ = other.move_history_;
        legal_moves_dirty_ = true;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
        is_terminal_cached_ = other.is_terminal_cached_;
        cached_result_ = other.cached_result_;
        terminal_check_dirty_ = other.terminal_check_dirty_;
    }
    return *this;
}

// Initialize the starting position
void ChessState::initializeStartingPosition() {
    initializeEmpty();
    
    // Set up white pieces
    setPiece(0, {PieceType::ROOK, PieceColor::WHITE});
    setPiece(1, {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(2, {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(3, {PieceType::QUEEN, PieceColor::WHITE});
    setPiece(4, {PieceType::KING, PieceColor::WHITE});
    setPiece(5, {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(6, {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(7, {PieceType::ROOK, PieceColor::WHITE});
    
    for (int i = 0; i < 8; ++i) {
        setPiece(8 + i, {PieceType::PAWN, PieceColor::WHITE});

    setPiece(56, {PieceType::ROOK, PieceColor::BLACK});
    setPiece(57, {PieceType::KNIGHT, PieceColor::BLACK});
    setPiece(58, {PieceType::BISHOP, PieceColor::BLACK});
    setPiece(59, {PieceType::QUEEN, PieceColor::BLACK});
    setPiece(60, {PieceType::KING, PieceColor::BLACK});
    setPiece(61, {PieceType::BISHOP, PieceColor::BLACK});
    setPiece(62, {PieceType::KNIGHT, PieceColor::BLACK});
    setPiece(63, {PieceType::ROOK, PieceColor::BLACK});
    
    for (int i = 0; i < 8; ++i) {
        setPiece(48 + i, {PieceType::PAWN, PieceColor::BLACK});
    }
    
    // Reset game state
    current_player_ = PieceColor::WHITE;
    castling_rights_ = {true, true, true, true};
    en_passant_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    move_history_.clear();
    
    // Mark caches as dirty
    invalidateCache();
}

void ChessState::initializeEmpty() {
    // Clear board
    for (int i = 0; i < NUM_SQUARES; ++i) {
        board_[i] = {PieceType::NONE, PieceColor::NONE};
    }
}

void ChessState::invalidateCache() {
    legal_moves_dirty_ = true;
    hash_dirty_ = true;
    terminal_check_dirty_ = true;
}

// FEN parsing
bool ChessState::setFromFEN(const std::string& fen) {
    initializeEmpty();
    
    std::istringstream ss(fen);
    std::string board, active_color, castling, en_passant, halfmove, fullmove;
    
    // Parse the board position
    if (!(ss >> board)) return false;
    
    int rank = 7;
    int file = 0;
    
    for (char c : board) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += c - '0';
        } else {
            Piece piece;
            if (c == 'P') { piece = {PieceType::PAWN, PieceColor::WHITE}; }
            else if (c == 'N') { piece = {PieceType::KNIGHT, PieceColor::WHITE}; }
            else if (c == 'B') { piece = {PieceType::BISHOP, PieceColor::WHITE}; }
            else if (c == 'R') { piece = {PieceType::ROOK, PieceColor::WHITE}; }
            else if (c == 'Q') { piece = {PieceType::QUEEN, PieceColor::WHITE}; }
            else if (c == 'K') { piece = {PieceType::KING, PieceColor::WHITE}; }
            else if (c == 'p') { piece = {PieceType::PAWN, PieceColor::BLACK}; }
            else if (c == 'n') { piece = {PieceType::KNIGHT, PieceColor::BLACK}; }
            else if (c == 'b') { piece = {PieceType::BISHOP, PieceColor::BLACK}; }
            else if (c == 'r') { piece = {PieceType::ROOK, PieceColor::BLACK}; }
            else if (c == 'q') { piece = {PieceType::QUEEN, PieceColor::BLACK}; }
            else if (c == 'k') { piece = {PieceType::KING, PieceColor::BLACK}; }
            else return false;
            
            setPiece(getSquare(rank, file), piece);
            file++;
        }
    }
    
    // Parse the active color
    if (!(ss >> active_color)) return false;
    current_player_ = (active_color == "w") ? PieceColor::WHITE : PieceColor::BLACK;
    
    // Parse castling availability
    if (!(ss >> castling)) return false;
    castling_rights_ = {false, false, false, false};
    for (char c : castling) {
        if (c == 'K') castling_rights_.white_kingside = true;
        else if (c == 'Q') castling_rights_.white_queenside = true;
        else if (c == 'k') castling_rights_.black_kingside = true;
        else if (c == 'q') castling_rights_.black_queenside = true;
    }
    
    // Parse en passant square
    if (!(ss >> en_passant)) return false;
    en_passant_square_ = (en_passant == "-") ? -1 : stringToSquare(en_passant);
    
    // Parse halfmove clock
    if (!(ss >> halfmove)) return false;
    halfmove_clock_ = std::stoi(halfmove);
    
    // Parse fullmove number
    if (!(ss >> fullmove)) return false;
    fullmove_number_ = std::stoi(fullmove);
    
    // Mark caches as dirty
    invalidateCache();
    
    return true;
}

// Generate FEN string from current position
std::string ChessState::toFEN() const {
    std::stringstream ss;
    
    // Board position
    for (int rank = 7; rank >= 0; --rank) {
        int emptyCount = 0;
        
        for (int file = 0; file < 8; ++file) {
            int square = getSquare(rank, file);
            Piece piece = board_[square];
            
            if (piece.is_empty()) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    ss << emptyCount;
                    emptyCount = 0;
                }
                
                char pieceChar;
                switch (piece.type) {
                    case PieceType::PAWN: pieceChar = 'P'; break;
                    case PieceType::KNIGHT: pieceChar = 'N'; break;
                    case PieceType::BISHOP: pieceChar = 'B'; break;
                    case PieceType::ROOK: pieceChar = 'R'; break;
                    case PieceType::QUEEN: pieceChar = 'Q'; break;
                    case PieceType::KING: pieceChar = 'K'; break;
                    default: pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::BLACK) {
                    pieceChar = std::tolower(pieceChar);
                }
                
                ss << pieceChar;
            }
        }
        
        if (emptyCount > 0) {
            ss << emptyCount;
        }
        
        if (rank > 0) {
            ss << '/';
        }
    }
    
    // Active color
    ss << ' ' << (current_player_ == PieceColor::WHITE ? 'w' : 'b');
    
    // Castling availability
    ss << ' ';
    if (!castling_rights_.white_kingside && !castling_rights_.white_queenside && 
        !castling_rights_.black_kingside && !castling_rights_.black_queenside) {
        ss << '-';
    } else {
        if (castling_rights_.white_kingside) ss << 'K';
        if (castling_rights_.white_queenside) ss << 'Q';
        if (castling_rights_.black_kingside) ss << 'k';
        if (castling_rights_.black_queenside) ss << 'q';
    }
    
    // En passant target square
    ss << ' ' << (en_passant_square_ == -1 ? "-" : squareToString(en_passant_square_));
    
    // Halfmove clock
    ss << ' ' << halfmove_clock_;
    
    // Fullmove number
    ss << ' ' << fullmove_number_;
    
    return ss.str();
}

// IGameState implementation
std::vector<int> ChessState::getLegalMoves() const {
    std::vector<ChessMove> chessMoves = generateLegalMoves();
    std::vector<int> actions;
    actions.reserve(chessMoves.size());
    
    for (const auto& move : chessMoves) {
        actions.push_back(chessMoveToAction(move));
    }
    
    return actions;
}

bool ChessState::isLegalMove(int action) const {
    ChessMove move = actionToChessMove(action);
    return isLegalMove(move);
}

void ChessState::makeMove(int action) {
    ChessMove move = actionToChessMove(action);
    makeMove(move);
}

bool ChessState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    
    // Get last move info
    MoveInfo info = move_history_.back();
    move_history_.pop_back();
    
    // Restore captured piece
    setPiece(info.move.to_square, {PieceType::NONE, PieceColor::NONE});
    setPiece(info.move.from_square, {PieceType::NONE, PieceColor::NONE});
    
    // Handle special moves
    if (info.was_castle) {
        // Undo castling rook move
        if (info.move.to_square > info.move.from_square) {
            // Kingside castle
            setPiece(info.move.from_square, {PieceType::KING, current_player_});
            setPiece(info.move.from_square + 3, {PieceType::ROOK, current_player_});
            setPiece(info.move.to_square, {PieceType::NONE, PieceColor::NONE});
            setPiece(info.move.from_square + 1, {PieceType::NONE, PieceColor::NONE});
        } else {
            // Queenside castle
            setPiece(info.move.from_square, {PieceType::KING, current_player_});
            setPiece(info.move.from_square - 4, {PieceType::ROOK, current_player_});
            setPiece(info.move.to_square, {PieceType::NONE, PieceColor::NONE});
            setPiece(info.move.from_square - 1, {PieceType::NONE, PieceColor::NONE});
        }
    } else if (info.was_en_passant) {
        // Restore captured pawn in en passant
        int pawn_square = info.move.to_square;
        if (current_player_ == PieceColor::WHITE) {
            pawn_square -= 8;
        } else {
            pawn_square += 8;
        }
        setPiece(pawn_square, {PieceType::PAWN, oppositeColor(current_player_)});
        setPiece(info.move.from_square, {PieceType::PAWN, current_player_});
    } else {
        // Normal move
        setPiece(info.move.from_square, getPiece(info.move.to_square));
        setPiece(info.move.to_square, info.captured_piece);
    }
    
    // Restore state
    castling_rights_ = info.castling_rights;
    en_passant_square_ = info.en_passant_square;
    halfmove_clock_ = info.halfmove_clock;
    
    // Switch player
    current_player_ = oppositeColor(current_player_);
    
    // Update fullmove counter
    if (current_player_ == PieceColor::WHITE) {
        fullmove_number_--;
    }
    
    invalidateCache();
    return true;
}

bool ChessState::isTerminal() const {
    if (!terminal_check_dirty_) {
        return is_terminal_cached_;
    }
    
    // Game is over if no legal moves (checkmate or stalemate)
    bool no_legal_moves = generateLegalMoves().empty();
    
    // Game is also over with 50-move rule, insufficient material, or threefold repetition
    bool fifty_move_rule = halfmove_clock_ >= 100;
    bool insufficient_material = hasInsufficientMaterial();
    bool threefold = isThreefoldRepetition();
    
    is_terminal_cached_ = no_legal_moves || fifty_move_rule || insufficient_material || threefold;
    
    if (is_terminal_cached_) {
        // Determine result
        if (no_legal_moves) {
            // If in check, it's checkmate
            if (isInCheck(current_player_)) {
                cached_result_ = (current_player_ == PieceColor::WHITE) ? 
                                 core::GameResult::WIN_PLAYER2 : core::GameResult::WIN_PLAYER1;
            } else {
                // Stalemate
                cached_result_ = core::GameResult::DRAW;
            }
        } else {
            // Draw by other rules
            cached_result_ = core::GameResult::DRAW;
        }
    } else {
        cached_result_ = core::GameResult::ONGOING;
    }
    
    terminal_check_dirty_ = false;
    return is_terminal_cached_;
}

core::GameResult ChessState::getGameResult() const {
    if (terminal_check_dirty_) {
        isTerminal();  // Updates cached_result_
    }
    return cached_result_;
}

int ChessState::getCurrentPlayer() const {
    return current_player_ == PieceColor::WHITE ? 1 : 2;
}

// Neural network representation
std::vector<std::vector<std::vector<float>>> ChessState::getTensorRepresentation() const {
    // Create output tensor with 12 planes (6 piece types * 2 colors)
    std::vector<std::vector<std::vector<float>>> tensor(12, 
        std::vector<std::vector<float>>(8, 
            std::vector<float>(8, 0.0f)));
    
    // Fill in piece positions
    for (int square = 0; square < 64; ++square) {
        int rank = getRank(square);
        int file = getFile(square);
        
        Piece piece = board_[square];
        if (piece.type != PieceType::NONE) {
            int plane = static_cast<int>(piece.type) - 1;
            if (piece.color == PieceColor::BLACK) {
                plane += 6;
            }
            tensor[plane][rank][file] = 1.0f;
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> ChessState::getEnhancedTensorRepresentation() const {
    // Start with basic piece planes
    std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
    
    // Add auxiliary planes
    // 13: Current player (1 for white, 0 for black)
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, current_player_ == PieceColor::WHITE ? 1.0f : 0.0f)));
    
    // 14-17: Castling rights
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, castling_rights_.white_kingside ? 1.0f : 0.0f)));
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, castling_rights_.white_queenside ? 1.0f : 0.0f)));
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, castling_rights_.black_kingside ? 1.0f : 0.0f)));
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, castling_rights_.black_queenside ? 1.0f : 0.0f)));
    
    // 18: En passant
    std::vector<std::vector<float>> en_passant_plane(8, std::vector<float>(8, 0.0f));
    if (en_passant_square_ != -1) {
        int rank = getRank(en_passant_square_);
        int file = getFile(en_passant_square_);
        en_passant_plane[rank][file] = 1.0f;
    }
    tensor.push_back(en_passant_plane);
    
    // 19: Halfmove clock normalized
    float halfmove_norm = std::min(1.0f, halfmove_clock_ / 100.0f);
    tensor.push_back(std::vector<std::vector<float>>(8, 
        std::vector<float>(8, halfmove_norm)));
    
    return tensor;
}

uint64_t ChessState::getHash() const {
    if (!hash_dirty_) {
        return hash_;
    }
    
    hash_ = 0;
    
    // Hash pieces
    for (int square = 0; square < 64; ++square) {
        Piece piece = board_[square];
        if (piece.type != PieceType::NONE) {
            int piece_idx = static_cast<int>(piece.type) - 1;
            if (piece.color == PieceColor::BLACK) {
                piece_idx += 6;
            }
            hash_ ^= zobrist_.getPieceHash(piece_idx, square);
        }
    }
    
    // Hash current player
    if (current_player_ == PieceColor::BLACK) {
        hash_ ^= zobrist_.getPlayerHash(0);
    }
    
    // Hash castling rights
    int castling_idx = (castling_rights_.white_kingside ? 1 : 0) |
                        (castling_rights_.white_queenside ? 2 : 0) |
                        (castling_rights_.black_kingside ? 4 : 0) |
                        (castling_rights_.black_queenside ? 8 : 0);
    hash_ ^= zobrist_.getFeatureHash(0, castling_idx);
    
    // Hash en passant
    if (en_passant_square_ != -1) {
        hash_ ^= zobrist_.getFeatureHash(1, en_passant_square_);
    }
    
    hash_dirty_ = false;
    return hash_;
}

std::unique_ptr<core::IGameState> ChessState::clone() const {
    return std::make_unique<ChessState>(*this);
}

// Convert between actions and chess moves
ChessMove ChessState::actionToChessMove(int action) const {
    int from_square = (action >> 12) & 0x3F;
    int to_square = (action >> 6) & 0x3F;
    int promotion = action & 0x7;
    
    PieceType promotion_piece = PieceType::NONE;
    if (promotion > 0) {
        promotion_piece = static_cast<PieceType>(promotion);
    }
    
    return {from_square, to_square, promotion_piece};
}

int ChessState::chessMoveToAction(const ChessMove& move) const {
    return (move.from_square << 12) | (move.to_square << 6) | static_cast<int>(move.promotion_piece);
}

std::string ChessState::actionToString(int action) const {
    ChessMove move = actionToChessMove(action);
    return moveToString(move);
}

std::optional<int> ChessState::stringToAction(const std::string& moveStr) const {
    auto move = stringToMove(moveStr);
    if (!move) {
        return std::nullopt;
    }
    return chessMoveToAction(*move);
}

std::string ChessState::moveToString(const ChessMove& move) const {
    std::string result = squareToString(move.from_square) + squareToString(move.to_square);
    
    if (move.promotion_piece != PieceType::NONE) {
        switch (move.promotion_piece) {
            case PieceType::KNIGHT: result += "n"; break;
            case PieceType::BISHOP: result += "b"; break;
            case PieceType::ROOK: result += "r"; break;
            case PieceType::QUEEN: result += "q"; break;
            default: break;
        }
    }
    
    return result;
}

std::optional<ChessMove> ChessState::stringToMove(const std::string& moveStr) const {
    if (moveStr.length() < 4 || moveStr.length() > 5) {
        return std::nullopt;
    }
    
    // Parse squares
    int from_square = stringToSquare(moveStr.substr(0, 2));
    int to_square = stringToSquare(moveStr.substr(2, 2));
    
    if (from_square == -1 || to_square == -1) {
        return std::nullopt;
    }
    
    // Parse promotion
    PieceType promotion_piece = PieceType::NONE;
    if (moveStr.length() == 5) {
        char promo = std::tolower(moveStr[4]);
        switch (promo) {
            case 'n': promotion_piece = PieceType::KNIGHT; break;
            case 'b': promotion_piece = PieceType::BISHOP; break;
            case 'r': promotion_piece = PieceType::ROOK; break;
            case 'q': promotion_piece = PieceType::QUEEN; break;
            default: return std::nullopt;
        }
    }
    
    return ChessMove{from_square, to_square, promotion_piece};
}

std::string ChessState::toString() const {
    std::stringstream ss;
    
    // Print board with rank and file labels
    ss << "  a b c d e f g h" << std::endl;
    for (int rank = 7; rank >= 0; --rank) {
        ss << (rank + 1) << " ";
        for (int file = 0; file < 8; ++file) {
            Piece piece = board_[getSquare(rank, file)];
            char pieceChar = '.';
            
            if (piece.type != PieceType::NONE) {
                switch (piece.type) {
                    case PieceType::PAWN: pieceChar = 'P'; break;
                    case PieceType::KNIGHT: pieceChar = 'N'; break;
                    case PieceType::BISHOP: pieceChar = 'B'; break;
                    case PieceType::ROOK: pieceChar = 'R'; break;
                    case PieceType::QUEEN: pieceChar = 'Q'; break;
                    case PieceType::KING: pieceChar = 'K'; break;
                    default: pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::BLACK) {
                    pieceChar = std::tolower(pieceChar);
                }
            }
            
            ss << pieceChar << " ";
        }
        ss << (rank + 1) << std::endl;
    }
    ss << "  a b c d e f g h" << std::endl;
    
    // Print additional information
    ss << "FEN: " << toFEN() << std::endl;
    ss << "Turn: " << (current_player_ == PieceColor::WHITE ? "White" : "Black") << std::endl;
    
    if (isInCheck(current_player_)) {
        ss << "Check!" << std::endl;
    }
    
    return ss.str();
}

bool ChessState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::CHESS) {
        return false;
    }
    
    try {
        const ChessState& otherChess = dynamic_cast<const ChessState&>(other);
        
        // Compare board positions
        for (int square = 0; square < 64; ++square) {
            if (board_[square] != otherChess.board_[square]) {
                return false;
            }
        }
        
        // Compare state
        return current_player_ == otherChess.current_player_ &&
               castling_rights_ == otherChess.castling_rights_ &&
               en_passant_square_ == otherChess.en_passant_square_ &&
               halfmove_clock_ == otherChess.halfmove_clock_ &&
               fullmove_number_ == otherChess.fullmove_number_;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> ChessState::getMoveHistory() const {
    std::vector<int> actionHistory;
    actionHistory.reserve(move_history_.size());
    
    for (const auto& info : move_history_) {
        actionHistory.push_back(chessMoveToAction(info.move));
    }
    
    return actionHistory;
}

// Check detection
bool ChessState::isInCheck(PieceColor color) const {
    if (color == PieceColor::NONE) {
        color = current_player_;
    }
    
    int king_square = getKingSquare(color);
    if (king_square == -1) {
        return false;  // No king found
    }
    
    return isSquareAttacked(king_square, oppositeColor(color));
}

// Square to string conversion
std::string ChessState::squareToString(int square) {
    if (square < 0 || square >= 64) {
        return "??";
    }
    
    int rank = getRank(square);
    int file = getFile(square);
    
    std::string result;
    result += static_cast<char>('a' + file);
    result += static_cast<char>('1' + rank);
    
    return result;
}

int ChessState::stringToSquare(const std::string& squareStr) {
    if (squareStr.length() != 2) {
        return -1;
    }
    
    char file_char = std::tolower(squareStr[0]);
    char rank_char = squareStr[1];
    
    if (file_char < 'a' || file_char > 'h' || rank_char < '1' || rank_char > '8') {
        return -1;
    }
    
    int file = file_char - 'a';
    int rank = rank_char - '1';
    
    return getSquare(rank, file);
}

// Implementation of legal move generation, check detection, etc. would follow...
// The complete implementation would be much larger.

} // namespace chess
} // namespace alphazero