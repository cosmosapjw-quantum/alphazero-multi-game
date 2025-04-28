// src/games/chess/chess_state.cpp
#include "alphazero/games/chess/chess_state.h"
#include "alphazero/games/chess/chess_rules.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <cmath>

namespace alphazero {
namespace chess {

// Constants for board representation
const int A1 = 56;
const int H1 = 63;
const int A8 = 0;
const int H8 = 7;
const int E1 = 60;
const int E8 = 4;

// Constructor
ChessState::ChessState(bool chess960, const std::string& fen)
    : IGameState(core::GameType::CHESS),
      chess960_(chess960),
      current_player_(PieceColor::WHITE),
      en_passant_square_(-1),
      halfmove_clock_(0),
      fullmove_number_(1),
      legal_moves_dirty_(true),
      hash_dirty_(true),
      terminal_check_dirty_(true),
      zobrist_(core::GameType::CHESS, 8, 12)  // 12 = 6 piece types * 2 colors
{
    // Initialize with empty board first
    initializeEmpty();
    
    // Initialize rules object
    rules_ = std::make_shared<ChessRules>(chess960_);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int square) { return this->getPiece(square); },
        [this](int square) { return ChessState::isValidSquare(square); },
        [this](PieceColor color) { return this->getKingSquare(color); }
    );
    
    if (!fen.empty()) {
        // If FEN is provided, use it
        if (!setFromFEN(fen)) {
            // If FEN parsing fails, fall back to starting position
            initializeStartingPosition();
        }
    } else {
        // Use standard starting position
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
      cached_legal_moves_(other.cached_legal_moves_),
      legal_moves_dirty_(other.legal_moves_dirty_),
      zobrist_(other.zobrist_),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_),
      is_terminal_cached_(other.is_terminal_cached_),
      cached_result_(other.cached_result_),
      terminal_check_dirty_(other.terminal_check_dirty_)
{
    // Initialize rules object
    rules_ = std::make_shared<ChessRules>(chess960_);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int square) { return this->getPiece(square); },
        [this](int square) { return ChessState::isValidSquare(square); },
        [this](PieceColor color) { return this->getKingSquare(color); }
    );
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
        cached_legal_moves_ = other.cached_legal_moves_;
        legal_moves_dirty_ = other.legal_moves_dirty_;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
        is_terminal_cached_ = other.is_terminal_cached_;
        cached_result_ = other.cached_result_;
        terminal_check_dirty_ = other.terminal_check_dirty_;
        
        // Reinitialize rules object
        rules_ = std::make_shared<ChessRules>(chess960_);
        
        // Set up board accessor functions for rules
        rules_->setBoardAccessor(
            [this](int square) { return this->getPiece(square); },
            [this](int square) { return ChessState::isValidSquare(square); },
            [this](PieceColor color) { return this->getKingSquare(color); }
        );
    }
    return *this;
}

// Initialize empty board
void ChessState::initializeEmpty() {
    // Initialize empty board
    for (int i = 0; i < NUM_SQUARES; ++i) {
        board_[i] = Piece();
    }
    
    // Reset game state
    current_player_ = PieceColor::WHITE;
    castling_rights_ = CastlingRights();
    en_passant_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    
    // Clear move history
    move_history_.clear();
    
    // Mark caches as dirty
    invalidateCache();
}

// Initialize standard starting position
void ChessState::initializeStartingPosition() {
    initializeEmpty();
    
    // Set up pawns
    for (int i = 0; i < 8; ++i) {
        // White pawns (rank 2)
        setPiece(getSquare(6, i), {PieceType::PAWN, PieceColor::WHITE});
        
        // Black pawns (rank 7)
        setPiece(getSquare(1, i), {PieceType::PAWN, PieceColor::BLACK});
    }
    
    // Set up rooks
    setPiece(A1, {PieceType::ROOK, PieceColor::WHITE});
    setPiece(H1, {PieceType::ROOK, PieceColor::WHITE});
    setPiece(A8, {PieceType::ROOK, PieceColor::BLACK});
    setPiece(H8, {PieceType::ROOK, PieceColor::BLACK});
    
    // Set up knights
    setPiece(getSquare(7, 1), {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(getSquare(7, 6), {PieceType::KNIGHT, PieceColor::WHITE});
    setPiece(getSquare(0, 1), {PieceType::KNIGHT, PieceColor::BLACK});
    setPiece(getSquare(0, 6), {PieceType::KNIGHT, PieceColor::BLACK});
    
    // Set up bishops
    setPiece(getSquare(7, 2), {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(getSquare(7, 5), {PieceType::BISHOP, PieceColor::WHITE});
    setPiece(getSquare(0, 2), {PieceType::BISHOP, PieceColor::BLACK});
    setPiece(getSquare(0, 5), {PieceType::BISHOP, PieceColor::BLACK});
    
    // Set up queens
    setPiece(getSquare(7, 3), {PieceType::QUEEN, PieceColor::WHITE});
    setPiece(getSquare(0, 3), {PieceType::QUEEN, PieceColor::BLACK});
    
    // Set up kings
    setPiece(E1, {PieceType::KING, PieceColor::WHITE});
    setPiece(E8, {PieceType::KING, PieceColor::BLACK});
    
    // Initialize castling rights
    castling_rights_.white_kingside = true;
    castling_rights_.white_queenside = true;
    castling_rights_.black_kingside = true;
    castling_rights_.black_queenside = true;
    
    // Update hash
    updateHash();
}

// Board manipulation methods
Piece ChessState::getPiece(int square) const {
    if (square < 0 || square >= NUM_SQUARES) {
        return Piece();
    }
    return board_[square];
}

void ChessState::setPiece(int square, const Piece& piece) {
    if (square < 0 || square >= NUM_SQUARES) {
        return;
    }
    board_[square] = piece;
    invalidateCache();
}

// Get/set methods for game state
CastlingRights ChessState::getCastlingRights() const {
    return castling_rights_;
}

int ChessState::getEnPassantSquare() const {
    return en_passant_square_;
}

int ChessState::getHalfmoveClock() const {
    return halfmove_clock_;
}

int ChessState::getFullmoveNumber() const {
    return fullmove_number_;
}

// FEN string conversion
std::string ChessState::toFEN() const {
    std::stringstream ss;
    
    // Board position
    for (int rank = 0; rank < 8; ++rank) {
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
                    case PieceType::PAWN:   pieceChar = 'p'; break;
                    case PieceType::KNIGHT: pieceChar = 'n'; break;
                    case PieceType::BISHOP: pieceChar = 'b'; break;
                    case PieceType::ROOK:   pieceChar = 'r'; break;
                    case PieceType::QUEEN:  pieceChar = 'q'; break;
                    case PieceType::KING:   pieceChar = 'k'; break;
                    default:                pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::WHITE) {
                    pieceChar = std::toupper(pieceChar);
                }
                
                ss << pieceChar;
            }
        }
        
        if (emptyCount > 0) {
            ss << emptyCount;
        }
        
        if (rank < 7) {
            ss << '/';
        }
    }
    
    // Active color
    ss << ' ' << (current_player_ == PieceColor::WHITE ? 'w' : 'b');
    
    // Castling availability
    ss << ' ';
    bool hasCastling = false;
    if (castling_rights_.white_kingside) {
        ss << 'K';
        hasCastling = true;
    }
    if (castling_rights_.white_queenside) {
        ss << 'Q';
        hasCastling = true;
    }
    if (castling_rights_.black_kingside) {
        ss << 'k';
        hasCastling = true;
    }
    if (castling_rights_.black_queenside) {
        ss << 'q';
        hasCastling = true;
    }
    
    if (!hasCastling) {
        ss << '-';
    }
    
    // En passant target square
    ss << ' ';
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        ss << squareToString(en_passant_square_);
    } else {
        ss << '-';
    }
    
    // Halfmove clock
    ss << ' ' << halfmove_clock_;
    
    // Fullmove number
    ss << ' ' << fullmove_number_;
    
    return ss.str();
}

bool ChessState::setFromFEN(const std::string& fen) {
    initializeEmpty();
    
    std::istringstream ss(fen);
    std::string boardPos, activeColor, castlingAvailability, enPassantTarget, halfmoveClock, fullmoveNumber;
    
    // Parse board position
    if (!(ss >> boardPos)) return false;
    
    int rank = 0, file = 0;
    for (char c : boardPos) {
        if (c == '/') {
            rank++;
            file = 0;
        } else if (std::isdigit(c)) {
            file += c - '0';
        } else {
            if (file >= 8 || rank >= 8) return false;
            
            Piece piece;
            char lowercase = std::tolower(c);
            
            switch (lowercase) {
                case 'p': piece.type = PieceType::PAWN; break;
                case 'n': piece.type = PieceType::KNIGHT; break;
                case 'b': piece.type = PieceType::BISHOP; break;
                case 'r': piece.type = PieceType::ROOK; break;
                case 'q': piece.type = PieceType::QUEEN; break;
                case 'k': piece.type = PieceType::KING; break;
                default: return false;
            }
            
            piece.color = std::isupper(c) ? PieceColor::WHITE : PieceColor::BLACK;
            setPiece(getSquare(rank, file), piece);
            file++;
        }
    }
    
    // Parse active color
    if (!(ss >> activeColor)) return false;
    current_player_ = (activeColor == "w") ? PieceColor::WHITE : PieceColor::BLACK;
    
    // Parse castling availability
    if (!(ss >> castlingAvailability)) return false;
    castling_rights_.white_kingside = castlingAvailability.find('K') != std::string::npos;
    castling_rights_.white_queenside = castlingAvailability.find('Q') != std::string::npos;
    castling_rights_.black_kingside = castlingAvailability.find('k') != std::string::npos;
    castling_rights_.black_queenside = castlingAvailability.find('q') != std::string::npos;
    
    // Parse en passant target square
    if (!(ss >> enPassantTarget)) return false;
    en_passant_square_ = (enPassantTarget == "-") ? -1 : stringToSquare(enPassantTarget);
    
    // Parse halfmove clock
    if (!(ss >> halfmoveClock)) return false;
    try {
        halfmove_clock_ = std::stoi(halfmoveClock);
    } catch (...) {
        return false;
    }
    
    // Parse fullmove number
    if (!(ss >> fullmoveNumber)) return false;
    try {
        fullmove_number_ = std::stoi(fullmoveNumber);
    } catch (...) {
        return false;
    }
    
    // Update hash and invalidate caches
    updateHash();
    
    return true;
}

// IGameState interface implementation
std::vector<int> ChessState::getLegalMoves() const {
    std::vector<int> result;
    
    std::vector<ChessMove> chessMovesLegal = generateLegalMoves();
    
    // Convert to action integers
    result.reserve(chessMovesLegal.size());
    for (const auto& move : chessMovesLegal) {
        result.push_back(chessMoveToAction(move));
    }
    
    return result;
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
    
    // Get the last move info
    const MoveInfo& moveInfo = move_history_.back();
    
    // Restore the captured piece
    setPiece(moveInfo.move.to_square, moveInfo.captured_piece);
    
    // Move the piece back
    Piece piece = getPiece(moveInfo.move.to_square);
    if (moveInfo.move.promotion_piece != PieceType::NONE) {
        // Restore the pawn
        piece.type = PieceType::PAWN;
    }
    setPiece(moveInfo.move.from_square, piece);
    setPiece(moveInfo.move.to_square, Piece());
    
    // Handle special moves
    if (moveInfo.was_castle) {
        // Move the rook back
        int kingFile = getFile(moveInfo.move.from_square);
        int rookFromFile = (kingFile < 4) ? 0 : 7;
        int rookToFile = (kingFile < 4) ? 3 : 5;
        
        int rookFromSquare = getSquare(getRank(moveInfo.move.from_square), rookFromFile);
        int rookToSquare = getSquare(getRank(moveInfo.move.from_square), rookToFile);
        
        Piece rook = getPiece(rookToSquare);
        setPiece(rookFromSquare, rook);
        setPiece(rookToSquare, Piece());
    } else if (moveInfo.was_en_passant) {
        // Restore the captured pawn
        int capturedPawnSquare = getSquare(getRank(moveInfo.move.from_square), getFile(moveInfo.move.to_square));
        setPiece(capturedPawnSquare, {PieceType::PAWN, oppositeColor(piece.color)});
    }
    
    // Restore game state
    current_player_ = oppositeColor(current_player_);
    castling_rights_ = moveInfo.castling_rights;
    en_passant_square_ = moveInfo.en_passant_square;
    halfmove_clock_ = moveInfo.halfmove_clock;
    
    if (current_player_ == PieceColor::WHITE) {
        fullmove_number_--;
    }
    
    // Remove the move from history
    move_history_.pop_back();
    
    // Invalidate caches
    invalidateCache();
    
    return true;
}

bool ChessState::isTerminal() const {
    if (!terminal_check_dirty_) {
        return is_terminal_cached_;
    }
    
    // Check if we have legal moves
    std::vector<ChessMove> legalMoves = generateLegalMoves();
    
    // If no legal moves, the game is over
    if (legalMoves.empty()) {
        is_terminal_cached_ = true;
        
        // If in check, it's checkmate; otherwise, stalemate
        if (isInCheck(current_player_)) {
            cached_result_ = current_player_ == PieceColor::WHITE ? 
                core::GameResult::WIN_PLAYER2 : core::GameResult::WIN_PLAYER1;
        } else {
            cached_result_ = core::GameResult::DRAW;
        }
        
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Check for draw by insufficient material
    if (rules_->hasInsufficientMaterial()) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Check for 50-move rule
    if (rules_->isFiftyMoveRule(halfmove_clock_)) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Create a vector of position hashes from move history
    std::vector<uint64_t> position_history;
    position_history.push_back(getHash());
    
    ChessState tempState(*this);
    
    for (int i = static_cast<int>(move_history_.size()) - 1; i >= 0; --i) {
        // Undo move
        tempState.undoMove();
        
        // Stop at irreversible moves
        if (tempState.getHalfmoveClock() == 0) {
            break;
        }
        
        // Add position hash
        position_history.push_back(tempState.getHash());
    }
    
    // Check for threefold repetition
    if (rules_->isThreefoldRepetition(position_history)) {
        is_terminal_cached_ = true;
        cached_result_ = core::GameResult::DRAW;
        terminal_check_dirty_ = false;
        return true;
    }
    
    // Game is not terminal
    is_terminal_cached_ = false;
    cached_result_ = core::GameResult::ONGOING;
    terminal_check_dirty_ = false;
    return false;
}

core::GameResult ChessState::getGameResult() const {
    if (terminal_check_dirty_) {
        isTerminal();  // This will update cached_result_
    }
    return cached_result_;
}

int ChessState::getCurrentPlayer() const {
    return static_cast<int>(current_player_);
}

std::vector<std::vector<std::vector<float>>> ChessState::getTensorRepresentation() const {
    // Basic 12-plane representation (6 piece types, 2 colors)
    std::vector<std::vector<std::vector<float>>> tensor(12, 
        std::vector<std::vector<float>>(8, 
            std::vector<float>(8, 0.0f)));
    
    // Fill tensor with piece positions
    for (int square = 0; square < NUM_SQUARES; ++square) {
        int rank = getRank(square);
        int file = getFile(square);
        Piece piece = board_[square];
        
        if (piece.type != PieceType::NONE) {
            int planeIdx = -1;
            if (piece.color == PieceColor::WHITE) {
                switch (piece.type) {
                    case PieceType::PAWN:   planeIdx = 0; break;
                    case PieceType::KNIGHT: planeIdx = 1; break;
                    case PieceType::BISHOP: planeIdx = 2; break;
                    case PieceType::ROOK:   planeIdx = 3; break;
                    case PieceType::QUEEN:  planeIdx = 4; break;
                    case PieceType::KING:   planeIdx = 5; break;
                    default: break;
                }
            } else {
                switch (piece.type) {
                    case PieceType::PAWN:   planeIdx = 6; break;
                    case PieceType::KNIGHT: planeIdx = 7; break;
                    case PieceType::BISHOP: planeIdx = 8; break;
                    case PieceType::ROOK:   planeIdx = 9; break;
                    case PieceType::QUEEN:  planeIdx = 10; break;
                    case PieceType::KING:   planeIdx = 11; break;
                    default: break;
                }
            }
            
            if (planeIdx >= 0) {
                tensor[planeIdx][rank][file] = 1.0f;
            }
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> ChessState::getEnhancedTensorRepresentation() const {
    // Start with basic representation
    std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
    
    // Add additional planes for enhanced features
    
    // 13: Current player (1 for white, 0 for black)
    tensor.push_back(std::vector<std::vector<float>>(8, 
                     std::vector<float>(8, current_player_ == PieceColor::WHITE ? 1.0f : 0.0f)));
    
    // 14: Castling rights
    std::vector<std::vector<float>> castlingPlane(8, std::vector<float>(8, 0.0f));
    float castlingValue = 0.0f;
    if (castling_rights_.white_kingside) castlingValue += 0.25f;
    if (castling_rights_.white_queenside) castlingValue += 0.25f;
    if (castling_rights_.black_kingside) castlingValue += 0.25f;
    if (castling_rights_.black_queenside) castlingValue += 0.25f;
    
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            castlingPlane[rank][file] = castlingValue;
        }
    }
    tensor.push_back(castlingPlane);
    
    // 15: En passant square
    std::vector<std::vector<float>> enPassantPlane(8, std::vector<float>(8, 0.0f));
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        int rank = getRank(en_passant_square_);
        int file = getFile(en_passant_square_);
        enPassantPlane[rank][file] = 1.0f;
    }
    tensor.push_back(enPassantPlane);
    
    // 16: Halfmove clock (normalized)
    float normalizedHalfmove = std::min(1.0f, static_cast<float>(halfmove_clock_) / 100.0f);
    tensor.push_back(std::vector<std::vector<float>>(8, 
                     std::vector<float>(8, normalizedHalfmove)));
    
    return tensor;
}

uint64_t ChessState::getHash() const {
    if (hash_dirty_) {
        updateHash();
    }
    return hash_;
}

std::unique_ptr<core::IGameState> ChessState::clone() const {
    return std::make_unique<ChessState>(*this);
}

std::string ChessState::actionToString(int action) const {
    ChessMove move = actionToChessMove(action);
    return moveToString(move);
}

std::optional<int> ChessState::stringToAction(const std::string& moveStr) const {
    std::optional<ChessMove> move = stringToMove(moveStr);
    if (!move) {
        return std::nullopt;
    }
    return chessMoveToAction(*move);
}

std::string ChessState::toString() const {
    std::stringstream ss;
    
    // Print board
    ss << "  a b c d e f g h" << std::endl;
    for (int rank = 0; rank < 8; ++rank) {
        ss << (8 - rank) << " ";
        for (int file = 0; file < 8; ++file) {
            int square = getSquare(rank, file);
            Piece piece = board_[square];
            
            if (piece.is_empty()) {
                ss << ". ";
            } else {
                char pieceChar;
                switch (piece.type) {
                    case PieceType::PAWN:   pieceChar = 'p'; break;
                    case PieceType::KNIGHT: pieceChar = 'n'; break;
                    case PieceType::BISHOP: pieceChar = 'b'; break;
                    case PieceType::ROOK:   pieceChar = 'r'; break;
                    case PieceType::QUEEN:  pieceChar = 'q'; break;
                    case PieceType::KING:   pieceChar = 'k'; break;
                    default:                pieceChar = '?'; break;
                }
                
                if (piece.color == PieceColor::WHITE) {
                    pieceChar = std::toupper(pieceChar);
                }
                
                ss << pieceChar << " ";
            }
        }
        ss << (8 - rank) << std::endl;
    }
    ss << "  a b c d e f g h" << std::endl;
    
    // Print current state
    ss << "Current player: " << (current_player_ == PieceColor::WHITE ? "White" : "Black") << std::endl;
    ss << "Castling rights: ";
    if (castling_rights_.white_kingside) ss << "K";
    if (castling_rights_.white_queenside) ss << "Q";
    if (castling_rights_.black_kingside) ss << "k";
    if (castling_rights_.black_queenside) ss << "q";
    if (!castling_rights_.white_kingside && !castling_rights_.white_queenside && 
        !castling_rights_.black_kingside && !castling_rights_.black_queenside) {
        ss << "-";
    }
    ss << std::endl;
    
    ss << "En passant square: ";
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        ss << squareToString(en_passant_square_);
    } else {
        ss << "-";
    }
    ss << std::endl;
    
    ss << "Halfmove clock: " << halfmove_clock_ << std::endl;
    ss << "Fullmove number: " << fullmove_number_ << std::endl;
    
    ss << "FEN: " << toFEN() << std::endl;
    
    return ss.str();
}

bool ChessState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::CHESS) {
        return false;
    }
    
    try {
        const ChessState& otherChess = dynamic_cast<const ChessState&>(other);
        
        // Compare board positions
        for (int square = 0; square < NUM_SQUARES; ++square) {
            if (board_[square] != otherChess.board_[square]) {
                return false;
            }
        }
        
        // Compare game state
        if (current_player_ != otherChess.current_player_ ||
            castling_rights_.white_kingside != otherChess.castling_rights_.white_kingside ||
            castling_rights_.white_queenside != otherChess.castling_rights_.white_queenside ||
            castling_rights_.black_kingside != otherChess.castling_rights_.black_kingside ||
            castling_rights_.black_queenside != otherChess.castling_rights_.black_queenside ||
            en_passant_square_ != otherChess.en_passant_square_ ||
            halfmove_clock_ != otherChess.halfmove_clock_ ||
            fullmove_number_ != otherChess.fullmove_number_) {
            return false;
        }
        
        return true;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> ChessState::getMoveHistory() const {
    std::vector<int> result;
    result.reserve(move_history_.size());
    
    for (const auto& moveInfo : move_history_) {
        result.push_back(chessMoveToAction(moveInfo.move));
    }
    
    return result;
}

bool ChessState::validate() const {
    // Check that there is exactly one king of each color
    int whiteKings = 0;
    int blackKings = 0;
    
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (piece.type == PieceType::KING) {
            if (piece.color == PieceColor::WHITE) {
                whiteKings++;
            } else if (piece.color == PieceColor::BLACK) {
                blackKings++;
            }
        }
    }
    
    if (whiteKings != 1 || blackKings != 1) {
        return false;
    }
    
    // Other validation checks could be added here
    
    return true;
}

// Move generation and validation methods
std::vector<ChessMove> ChessState::generateLegalMoves() const {
    if (!legal_moves_dirty_) {
        return cached_legal_moves_;
    }
    
    // Generate legal moves using rules
    cached_legal_moves_ = rules_->generateLegalMoves(
        current_player_, castling_rights_, en_passant_square_);
    legal_moves_dirty_ = false;
    
    return cached_legal_moves_;
}

bool ChessState::isLegalMove(const ChessMove& move) const {
    return rules_->isLegalMove(move, current_player_, castling_rights_, en_passant_square_);
}

void ChessState::makeMove(const ChessMove& move) {
    if (!isLegalMove(move)) {
        throw std::runtime_error("Illegal move attempted");
    }
    
    // Store move info for undoing
    MoveInfo moveInfo;
    moveInfo.move = move;
    moveInfo.captured_piece = getPiece(move.to_square);
    moveInfo.castling_rights = castling_rights_;
    moveInfo.en_passant_square = en_passant_square_;
    moveInfo.halfmove_clock = halfmove_clock_;
    moveInfo.was_castle = false;
    moveInfo.was_en_passant = false;
    
    // Get the moving piece
    Piece piece = getPiece(move.from_square);
    
    // Update halfmove clock
    if (piece.type == PieceType::PAWN || !moveInfo.captured_piece.is_empty()) {
        // Pawn move or capture resets the clock
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }
    
    // Clear en passant target
    int old_ep_square = en_passant_square_;
    en_passant_square_ = -1;
    
    // Handle special pawn moves
    if (piece.type == PieceType::PAWN) {
        int fromRank = getRank(move.from_square);
        int toRank = getRank(move.to_square);
        int fromFile = getFile(move.from_square);
        int toFile = getFile(move.to_square);
        
        // Check for two-square pawn move (set en passant target)
        if (std::abs(fromRank - toRank) == 2) {
            int epRank = (fromRank + toRank) / 2;
            en_passant_square_ = getSquare(epRank, fromFile);
        }
        
        // Check for en passant capture
        if (move.to_square == old_ep_square) {
            int capturedPawnRank = getRank(move.from_square);
            int capturedPawnFile = getFile(move.to_square);
            int capturedPawnSquare = getSquare(capturedPawnRank, capturedPawnFile);
            
            // Remove the captured pawn
            setPiece(capturedPawnSquare, Piece());
            moveInfo.was_en_passant = true;
        }
        
        // Handle promotion
        if (move.promotion_piece != PieceType::NONE) {
            piece.type = move.promotion_piece;
        }
    }
    
    // Handle castling
    if (piece.type == PieceType::KING && std::abs(getFile(move.from_square) - getFile(move.to_square)) == 2) {
        int rank = getRank(move.from_square);
        bool isKingside = getFile(move.to_square) > getFile(move.from_square);
        
        // Move the rook too
        if (isKingside) {
            // Kingside castling
            int rookFrom = getSquare(rank, 7);
            int rookTo = getSquare(rank, 5);
            Piece rook = getPiece(rookFrom);
            setPiece(rookFrom, Piece());
            setPiece(rookTo, rook);
        } else {
            // Queenside castling
            int rookFrom = getSquare(rank, 0);
            int rookTo = getSquare(rank, 3);
            Piece rook = getPiece(rookFrom);
            setPiece(rookFrom, Piece());
            setPiece(rookTo, rook);
        }
        
        moveInfo.was_castle = true;
    }
    
    // Update castling rights
    castling_rights_ = rules_->getUpdatedCastlingRights(
        move, piece, moveInfo.captured_piece, castling_rights_);
    
    // Move the piece
    setPiece(move.from_square, Piece());
    setPiece(move.to_square, piece);
    
    // Switch players
    current_player_ = oppositeColor(current_player_);
    
    // Update fullmove number
    if (current_player_ == PieceColor::WHITE) {
        fullmove_number_++;
    }
    
    // Add to move history
    move_history_.push_back(moveInfo);
    
    // Invalidate caches
    invalidateCache();
}

// Check and check detection
bool ChessState::isInCheck(PieceColor color) const {
    if (color == PieceColor::NONE) {
        color = current_player_;
    }
    
    return rules_->isInCheck(color);
}

bool ChessState::isSquareAttacked(int square, PieceColor by_color) const {
    return rules_->isSquareAttacked(square, by_color);
}

// Utility methods
int ChessState::getKingSquare(PieceColor color) const {
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (piece.type == PieceType::KING && piece.color == color) {
            return square;
        }
    }
    return -1;  // King not found
}

void ChessState::invalidateCache() {
    legal_moves_dirty_ = true;
    hash_dirty_ = true;
    terminal_check_dirty_ = true;
}

void ChessState::updateHash() const {
    hash_ = 0;
    
    // Hash board position
    for (int square = 0; square < NUM_SQUARES; ++square) {
        Piece piece = board_[square];
        if (!piece.is_empty()) {
            int pieceIdx = static_cast<int>(piece.type) - 1;
            if (piece.color == PieceColor::BLACK) {
                pieceIdx += 6;  // Offset for black pieces
            }
            hash_ ^= zobrist_.getPieceHash(pieceIdx, square);
        }
    }
    
    // Hash current player
    if (current_player_ == PieceColor::BLACK) {
        hash_ ^= zobrist_.getPlayerHash(1);
    }
    
    // Hash castling rights
    int castlingIdx = 0;
    if (castling_rights_.white_kingside) castlingIdx |= 1;
    if (castling_rights_.white_queenside) castlingIdx |= 2;
    if (castling_rights_.black_kingside) castlingIdx |= 4;
    if (castling_rights_.black_queenside) castlingIdx |= 8;
    hash_ ^= zobrist_.getFeatureHash(0, castlingIdx);
    
    // Hash en passant square
    if (en_passant_square_ >= 0 && en_passant_square_ < NUM_SQUARES) {
        hash_ ^= zobrist_.getFeatureHash(1, en_passant_square_);
    }
    
    hash_dirty_ = false;
}

// Move <-> Action conversion
ChessMove ChessState::actionToChessMove(int action) const {
    int fromSquare = (action >> 6) & 0x3F;
    int toSquare = action & 0x3F;
    int promotionCode = (action >> 12) & 0x7;
    
    PieceType promotionPiece = PieceType::NONE;
    switch (promotionCode) {
        case 1: promotionPiece = PieceType::QUEEN; break;
        case 2: promotionPiece = PieceType::ROOK; break;
        case 3: promotionPiece = PieceType::BISHOP; break;
        case 4: promotionPiece = PieceType::KNIGHT; break;
        default: promotionPiece = PieceType::NONE; break;
    }
    
    return {fromSquare, toSquare, promotionPiece};
}

int ChessState::chessMoveToAction(const ChessMove& move) const {
    int fromSquare = move.from_square & 0x3F;
    int toSquare = move.to_square & 0x3F;
    
    int promotionCode = 0;
    switch (move.promotion_piece) {
        case PieceType::QUEEN:  promotionCode = 1; break;
        case PieceType::ROOK:   promotionCode = 2; break;
        case PieceType::BISHOP: promotionCode = 3; break;
        case PieceType::KNIGHT: promotionCode = 4; break;
        default: promotionCode = 0; break;
    }
    
    return (promotionCode << 12) | (fromSquare << 6) | toSquare;
}

// String conversion utilities
std::string ChessState::squareToString(int square) {
    if (square < 0 || square >= 64) {
        return "";
    }
    
    int rank = getRank(square);
    int file = getFile(square);
    
    char fileChar = 'a' + file;
    char rankChar = '8' - rank;
    
    return std::string({fileChar, rankChar});
}

int ChessState::stringToSquare(const std::string& squareStr) {
    if (squareStr.length() != 2) {
        return -1;
    }
    
    char fileChar = squareStr[0];
    char rankChar = squareStr[1];
    
    if (fileChar < 'a' || fileChar > 'h' || rankChar < '1' || rankChar > '8') {
        return -1;
    }
    
    int file = fileChar - 'a';
    int rank = '8' - rankChar;
    
    return getSquare(rank, file);
}

std::string ChessState::moveToString(const ChessMove& move) const {
    std::string result = squareToString(move.from_square) + squareToString(move.to_square);
    
    // Add promotion piece
    if (move.promotion_piece != PieceType::NONE) {
        char promotionChar = ' ';
        switch (move.promotion_piece) {
            case PieceType::QUEEN:  promotionChar = 'q'; break;
            case PieceType::ROOK:   promotionChar = 'r'; break;
            case PieceType::BISHOP: promotionChar = 'b'; break;
            case PieceType::KNIGHT: promotionChar = 'n'; break;
            default: break;
        }
        
        if (promotionChar != ' ') {
            result += promotionChar;
        }
    }
    
    return result;
}

std::optional<ChessMove> ChessState::stringToMove(const std::string& moveStr) const {
    // Parse algebraic notation
    if (moveStr.length() < 4) {
        return std::nullopt;
    }
    
    int fromSquare = stringToSquare(moveStr.substr(0, 2));
    int toSquare = stringToSquare(moveStr.substr(2, 2));
    
    if (fromSquare == -1 || toSquare == -1) {
        return std::nullopt;
    }
    
    // Check for promotion
    PieceType promotionPiece = PieceType::NONE;
    if (moveStr.length() >= 5) {
        char promotionChar = std::tolower(moveStr[4]);
        switch (promotionChar) {
            case 'q': promotionPiece = PieceType::QUEEN; break;
            case 'r': promotionPiece = PieceType::ROOK; break;
            case 'b': promotionPiece = PieceType::BISHOP; break;
            case 'n': promotionPiece = PieceType::KNIGHT; break;
            default: break;
        }
    }
    
    return ChessMove{fromSquare, toSquare, promotionPiece};
}

std::string ChessState::toSAN(const ChessMove& move) const {
    // Implementation of Standard Algebraic Notation (SAN)
    Piece piece = getPiece(move.from_square);
    if (piece.is_empty()) {
        return "";
    }
    
    std::string san;
    
    // Castling
    if (piece.type == PieceType::KING && 
        std::abs(getFile(move.from_square) - getFile(move.to_square)) == 2) {
        if (getFile(move.to_square) > getFile(move.from_square)) {
            return "O-O";  // Kingside castling
        } else {
            return "O-O-O";  // Queenside castling
        }
    }
    
    // Piece letter (except for pawns)
    if (piece.type != PieceType::PAWN) {
        char pieceChar = ' ';
        switch (piece.type) {
            case PieceType::KNIGHT: pieceChar = 'N'; break;
            case PieceType::BISHOP: pieceChar = 'B'; break;
            case PieceType::ROOK:   pieceChar = 'R'; break;
            case PieceType::QUEEN:  pieceChar = 'Q'; break;
            case PieceType::KING:   pieceChar = 'K'; break;
            default: break;
        }
        san += pieceChar;
    }
    
    // Disambiguation
    std::vector<ChessMove> legalMoves = generateLegalMoves();
    std::vector<ChessMove> ambiguousMoves;
    
    for (const auto& m : legalMoves) {
        if (m.to_square == move.to_square && m.from_square != move.from_square) {
            Piece p = getPiece(m.from_square);
            if (p.type == piece.type && p.color == piece.color) {
                ambiguousMoves.push_back(m);
            }
        }
    }
    
    if (!ambiguousMoves.empty()) {
        bool sameFile = false;
        bool sameRank = false;
        
        for (const auto& m : ambiguousMoves) {
            if (getFile(m.from_square) == getFile(move.from_square)) {
                sameFile = true;
            }
            if (getRank(m.from_square) == getRank(move.from_square)) {
                sameRank = true;
            }
        }
        
        if (!sameFile) {
            // Disambiguate by file
            san += 'a' + getFile(move.from_square);
        } else if (!sameRank) {
            // Disambiguate by rank
            san += '8' - getRank(move.from_square);
        } else {
            // Disambiguate by both
            san += squareToString(move.from_square);
        }
    } else if (piece.type == PieceType::PAWN && getFile(move.from_square) != getFile(move.to_square)) {
        // Pawn capture: indicate file of origin
        san += 'a' + getFile(move.from_square);
    }
    
    // Capture symbol
    Piece capturedPiece = getPiece(move.to_square);
    bool isCapture = !capturedPiece.is_empty();
    
    // Special case: en passant capture
    if (piece.type == PieceType::PAWN && move.to_square == en_passant_square_) {
        isCapture = true;
    }
    
    if (isCapture) {
        san += "x";
    }
    
    // Destination square
    san += squareToString(move.to_square);
    
    // Promotion
    if (move.promotion_piece != PieceType::NONE) {
        san += "=";
        char promotionChar = ' ';
        switch (move.promotion_piece) {
            case PieceType::QUEEN:  promotionChar = 'Q'; break;
            case PieceType::ROOK:   promotionChar = 'R'; break;
            case PieceType::BISHOP: promotionChar = 'B'; break;
            case PieceType::KNIGHT: promotionChar = 'N'; break;
            default: break;
        }
        san += promotionChar;
    }
    
    // Check and checkmate
    ChessState tempState(*this);
    tempState.makeMove(move);
    
    if (tempState.isInCheck(oppositeColor(piece.color))) {
        // Generate legal moves for the opponent after this move
        tempState.legal_moves_dirty_ = true;
        std::vector<ChessMove> opponentMoves = tempState.generateLegalMoves();
        
        if (opponentMoves.empty()) {
            san += "#";  // Checkmate
        } else {
            san += "+";  // Check
        }
    }
    
    return san;
}

std::optional<ChessMove> ChessState::fromSAN(const std::string& sanStr) const {
    // This is a simplified SAN parser, not handling all edge cases
    if (sanStr.empty()) {
        return std::nullopt;
    }
    
    // Castling
    if (sanStr == "O-O" || sanStr == "0-0") {
        // Kingside castling
        int rank = (current_player_ == PieceColor::WHITE) ? 7 : 0;
        int kingSquare = getSquare(rank, 4);
        int targetSquare = getSquare(rank, 6);
        return ChessMove{kingSquare, targetSquare};
    } else if (sanStr == "O-O-O" || sanStr == "0-0-0") {
        // Queenside castling
        int rank = (current_player_ == PieceColor::WHITE) ? 7 : 0;
        int kingSquare = getSquare(rank, 4);
        int targetSquare = getSquare(rank, 2);
        return ChessMove{kingSquare, targetSquare};
    }
    
    // Normal move
    size_t i = 0;
    PieceType pieceType = PieceType::PAWN;
    
    // Piece type
    if (std::isupper(sanStr[0])) {
        switch (sanStr[0]) {
            case 'N': pieceType = PieceType::KNIGHT; break;
            case 'B': pieceType = PieceType::BISHOP; break;
            case 'R': pieceType = PieceType::ROOK; break;
            case 'Q': pieceType = PieceType::QUEEN; break;
            case 'K': pieceType = PieceType::KING; break;
            default: return std::nullopt;
        }
        i++;
    }
    
    // Source disambiguation (file and/or rank)
    int fromFile = -1;
    int fromRank = -1;
    
    if (i < sanStr.length() && sanStr[i] >= 'a' && sanStr[i] <= 'h') {
        fromFile = sanStr[i] - 'a';
        i++;
    }
    
    if (i < sanStr.length() && sanStr[i] >= '1' && sanStr[i] <= '8') {
        fromRank = '8' - sanStr[i];
        i++;
    }
    
    // Capture
    if (i < sanStr.length() && sanStr[i] == 'x') {
        i++;
    }
    
    // Destination square
    if (i + 1 >= sanStr.length()) {
        return std::nullopt;
    }
    
    if (sanStr[i] < 'a' || sanStr[i] > 'h' || 
        sanStr[i+1] < '1' || sanStr[i+1] > '8') {
        return std::nullopt;
    }
    
    int toFile = sanStr[i] - 'a';
    int toRank = '8' - sanStr[i+1];
    int toSquare = getSquare(toRank, toFile);
    i += 2;
    
    // Promotion
    PieceType promotionPiece = PieceType::NONE;
    if (i + 1 < sanStr.length() && sanStr[i] == '=') {
        i++;
        switch (sanStr[i]) {
            case 'Q': promotionPiece = PieceType::QUEEN; break;
            case 'R': promotionPiece = PieceType::ROOK; break;
            case 'B': promotionPiece = PieceType::BISHOP; break;
            case 'N': promotionPiece = PieceType::KNIGHT; break;
            default: return std::nullopt;
        }
        i++;
    }
    
    // Find matching legal move
    auto legalMoves = generateLegalMoves();
    
    for (const auto& move : legalMoves) {
        Piece piece = getPiece(move.from_square);
        
        if (piece.type != pieceType || move.to_square != toSquare) {
            continue;
        }
        
        if ((fromFile != -1 && getFile(move.from_square) != fromFile) ||
            (fromRank != -1 && getRank(move.from_square) != fromRank)) {
            continue;
        }
        
        if (promotionPiece != PieceType::NONE && move.promotion_piece != promotionPiece) {
            continue;
        }
        
        return move;
    }
    
    return std::nullopt;
}

} // namespace chess
} // namespace alphazero