#!/usr/bin/env python3
"""
REST API server for AlphaZero Multi-Game AI Engine.

This script provides a RESTful API for interfacing with the AlphaZero engine,
allowing external systems to create games, make moves, and get AI responses.

Usage:
    python api_server.py [options]

Options:
    --host HOST             Host to bind the server to (default: 127.0.0.1)
    --port PORT             Port to bind the server to (default: 5000)
    --model-dir DIR         Directory containing models (default: models)
    --simulations SIMS      Number of MCTS simulations per move (default: 800)
    --threads THREADS       Number of threads (default: 4)
    --temperature TEMP      Temperature for move selection (default: 0.1)
    --debug                 Enable debug mode
    --api-key KEY           API key for authentication (optional)
    --log-file FILE         Path to log file (optional)
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union

import _alphazero_cpp as az

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
except ImportError:
    print("Error: Flask not installed. Install with 'pip install flask flask-cors'")
    sys.exit(1)


class GameSession:
    """Game session for managing a single game."""
    
    def __init__(self, game_type: az.GameType, board_size: int, variant_rules: bool,
                 nn: Optional[az.NeuralNetwork] = None, simulations: int = 800,
                 threads: int = 4, temperature: float = 0.1):
        """Initialize a game session."""
        self.id = str(uuid.uuid4())
        self.game_type = game_type
        self.board_size = board_size
        self.variant_rules = variant_rules
        self.game_state = az.createGameState(game_type, board_size, variant_rules)
        self.nn = nn
        self.simulations = simulations
        self.threads = threads
        self.temperature = temperature
        self.tt = az.TranspositionTable(1048576, 1024)
        self.mcts = az.ParallelMCTS(
            self.game_state, self.nn, self.tt, 
            self.threads, self.simulations
        )
        self.mcts.setCPuct(1.5)
        self.mcts.setFpuReduction(0.0)
        self.last_activity = time.time()
        self.lock = threading.Lock()
    
    def make_move(self, action: int) -> bool:
        """Make a move in the game."""
        with self.lock:
            self.last_activity = time.time()
            
            try:
                # Check if move is legal
                if not self.game_state.isLegalMove(action):
                    return False
                
                # Make the move
                self.game_state.makeMove(action)
                
                # Update MCTS tree
                self.mcts.updateWithMove(action)
                
                return True
            except Exception as e:
                logging.error(f"Error making move: {e}")
                return False
    
    def get_ai_move(self) -> Optional[int]:
        """Get an AI move."""
        with self.lock:
            self.last_activity = time.time()
            
            if self.game_state.isTerminal():
                return None
            
            try:
                # Run search
                self.mcts.search()
                
                # Select action
                action = self.mcts.selectAction(False, self.temperature)
                
                # Make the move
                if action >= 0 and self.game_state.isLegalMove(action):
                    self.game_state.makeMove(action)
                    self.mcts.updateWithMove(action)
                    return action
                else:
                    return None
            except Exception as e:
                logging.error(f"Error getting AI move: {e}")
                return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current game state."""
        with self.lock:
            self.last_activity = time.time()
            
            game_result = self.game_state.getGameResult()
            result_map = {
                az.GameResult.ONGOING: "ongoing",
                az.GameResult.DRAW: "draw",
                az.GameResult.WIN_PLAYER1: "player1_win",
                az.GameResult.WIN_PLAYER2: "player2_win"
            }
            
            # Get legal moves
            legal_moves = self.game_state.getLegalMoves()
            
            # Get policy from MCTS if available
            policy = None
            if not self.game_state.isTerminal():
                # Simple search to get policy
                simulations_backup = self.mcts.getNumSimulations()
                self.mcts.setNumSimulations(100)
                self.mcts.search()
                self.mcts.setNumSimulations(simulations_backup)
                policy = self.mcts.getActionProbabilities(0.1)
            
            return {
                "id": self.id,
                "game_type": str(self.game_type).split(".")[-1],
                "board_size": self.board_size,
                "variant_rules": self.variant_rules,
                "current_player": self.game_state.getCurrentPlayer(),
                "result": result_map.get(game_result, "unknown"),
                "is_terminal": self.game_state.isTerminal(),
                "legal_moves": legal_moves,
                "board": self.get_board_representation(),
                "last_move": self.game_state.getMoveHistory()[-1] if self.game_state.getMoveHistory() else None,
                "move_history": self.game_state.getMoveHistory(),
                "policy": policy
            }
    
    def get_board_representation(self) -> List[List[int]]:
        """Get a 2D representation of the board."""
        if self.game_type == az.GameType.GOMOKU:
            # Gomoku state has a get_board method
            if hasattr(self.game_state, 'get_board'):
                return self.game_state.get_board()
            
            # Fallback: Create a board from tensor representation
            tensor = self.game_state.getTensorRepresentation()
            board_size = self.board_size
            board = [[0 for _ in range(board_size)] for _ in range(board_size)]
            
            # First plane is typically player 1 stones
            # Second plane is typically player 2 stones
            for y in range(board_size):
                for x in range(board_size):
                    if tensor[0][y][x] > 0.5:
                        board[y][x] = 1
                    elif tensor[1][y][x] > 0.5:
                        board[y][x] = 2
            
            return board
        
        elif self.game_type == az.GameType.CHESS:
            # For chess, return FEN string if available
            if hasattr(self.game_state, 'get_fen'):
                return self.game_state.get_fen()
            
            # Fallback: Return a simple representation
            tensor = self.game_state.getTensorRepresentation()
            # First 6 planes are typically white pieces (pawn, knight, bishop, rook, queen, king)
            # Next 6 planes are typically black pieces
            board = [[0 for _ in range(8)] for _ in range(8)]
            
            for y in range(8):
                for x in range(8):
                    for p in range(6):
                        if tensor[p][y][x] > 0.5:
                            board[y][x] = p + 1  # White pieces: 1-6
                        elif tensor[p+6][y][x] > 0.5:
                            board[y][x] = -(p + 1)  # Black pieces: -1 to -6
            
            return board
        
        elif self.game_type == az.GameType.GO:
            # Go state usually has a get_board method
            if hasattr(self.game_state, 'get_board'):
                return self.game_state.get_board()
            
            # Fallback: Create a board from tensor representation
            tensor = self.game_state.getTensorRepresentation()
            board_size = self.board_size
            board = [[0 for _ in range(board_size)] for _ in range(board_size)]
            
            # First plane is typically black stones
            # Second plane is typically white stones
            for y in range(board_size):
                for x in range(board_size):
                    if tensor[0][y][x] > 0.5:
                        board[y][x] = 1
                    elif tensor[1][y][x] > 0.5:
                        board[y][x] = 2
            
            return board
        
        # Default case
        return []


class SessionManager:
    """Manager for game sessions."""
    
    def __init__(self, model_dir: str, cleanup_interval: int = 3600, session_timeout: int = 3600*24):
        """Initialize the session manager."""
        self.sessions: Dict[str, GameSession] = {}
        self.model_dir = model_dir
        self.neural_networks: Dict[str, az.NeuralNetwork] = {}
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout
        self.lock = threading.Lock()
        
        # Load neural networks
        self._load_neural_networks()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def _load_neural_networks(self):
        """Load neural networks from model directory."""
        if not os.path.exists(self.model_dir):
            logging.warning(f"Model directory {self.model_dir} does not exist")
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith((".pt", ".pth")):
                # Try to determine game type from filename
                game_type = None
                if "gomoku" in filename.lower():
                    game_type = az.GameType.GOMOKU
                    board_size = 15
                elif "chess" in filename.lower():
                    game_type = az.GameType.CHESS
                    board_size = 8
                elif "go" in filename.lower():
                    if "9x9" in filename.lower():
                        game_type = az.GameType.GO
                        board_size = 9
                    elif "19x19" in filename.lower():
                        game_type = az.GameType.GO
                        board_size = 19
                    else:
                        game_type = az.GameType.GO
                        board_size = 19
                
                if game_type is not None:
                    try:
                        model_path = os.path.join(self.model_dir, filename)
                        nn = az.createNeuralNetwork(model_path, game_type, board_size, True)
                        if nn:
                            # Use filename without extension as the key
                            key = os.path.splitext(filename)[0]
                            self.neural_networks[key] = nn
                            logging.info(f"Loaded neural network from {filename}")
                    except Exception as e:
                        logging.error(f"Failed to load neural network from {filename}: {e}")
    
    def _cleanup_sessions(self):
        """Clean up inactive sessions."""
        while True:
            time.sleep(self.cleanup_interval)
            current_time = time.time()
            
            with self.lock:
                # Find sessions to remove
                to_remove = []
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        to_remove.append(session_id)
                
                # Remove sessions
                for session_id in to_remove:
                    del self.sessions[session_id]
                    logging.info(f"Cleaned up inactive session {session_id}")
    
    def create_session(self, game_type: az.GameType, board_size: int, variant_rules: bool,
                      model_name: Optional[str] = None, simulations: int = 800,
                      threads: int = 4, temperature: float = 0.1) -> str:
        """Create a new game session."""
        with self.lock:
            # Get neural network if specified
            nn = None
            if model_name:
                nn = self.neural_networks.get(model_name)
                if not nn:
                    # Try loading as a path
                    if os.path.exists(model_name) and (model_name.endswith(".pt") or model_name.endswith(".pth")):
                        try:
                            nn = az.createNeuralNetwork(model_name, game_type, board_size, True)
                        except Exception as e:
                            logging.error(f"Failed to load neural network from {model_name}: {e}")
            
            # Create session
            session = GameSession(
                game_type, board_size, variant_rules,
                nn, simulations, threads, temperature
            )
            
            # Add to sessions
            self.sessions[session.id] = session
            
            return session.id
    
    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get a game session by ID."""
        with self.lock:
            return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a game session."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get a list of available models."""
        with self.lock:
            return [
                {
                    "name": name,
                    "device_info": nn.getDeviceInfo() if hasattr(nn, "getDeviceInfo") else "Unknown"
                }
                for name, nn in self.neural_networks.items()
            ]


# Create Flask app
app = Flask(__name__)
CORS(app)

# Global session manager
session_manager = None

# API key
api_key = None


@app.before_request
def check_api_key():
    """Check API key if required."""
    if api_key and request.endpoint != 'health_check':
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {api_key}":
            abort(401, description="Invalid API key")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/games', methods=['POST'])
def create_game():
    """Create a new game."""
    data = request.json
    
    # Extract parameters
    game_type_str = data.get('game_type', 'GOMOKU').upper()
    board_size = data.get('board_size', 0)
    variant_rules = data.get('variant_rules', False)
    model_name = data.get('model', None)
    simulations = data.get('simulations', 800)
    threads = data.get('threads', 4)
    temperature = data.get('temperature', 0.1)
    
    # Map game type string to enum
    game_type_map = {
        'GOMOKU': az.GameType.GOMOKU,
        'CHESS': az.GameType.CHESS,
        'GO': az.GameType.GO
    }
    
    if game_type_str not in game_type_map:
        return jsonify({"error": f"Invalid game type: {game_type_str}"}), 400
    
    game_type = game_type_map[game_type_str]
    
    # Default board sizes
    if board_size <= 0:
        if game_type == az.GameType.GOMOKU:
            board_size = 15
        elif game_type == az.GameType.CHESS:
            board_size = 8
        elif game_type == az.GameType.GO:
            board_size = 19
        else:
            board_size = 15
    
    # Create session
    session_id = session_manager.create_session(
        game_type, board_size, variant_rules,
        model_name, simulations, threads, temperature
    )
    
    # Get initial state
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Failed to create game session"}), 500
    
    # Return game info
    return jsonify({
        "session_id": session_id,
        "state": session.get_state()
    })


@app.route('/api/games/<session_id>', methods=['GET'])
def get_game(session_id):
    """Get game information."""
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Game session not found"}), 404
    
    return jsonify(session.get_state())


@app.route('/api/games/<session_id>', methods=['DELETE'])
def delete_game(session_id):
    """Delete a game."""
    success = session_manager.delete_session(session_id)
    if not success:
        return jsonify({"error": "Game session not found"}), 404
    
    return jsonify({"success": True})


@app.route('/api/games/<session_id>/moves', methods=['POST'])
def make_move(session_id):
    """Make a move in the game."""
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Game session not found"}), 404
    
    data = request.json
    
    # Extract parameters
    action = data.get('action')
    move_str = data.get('move')
    
    if action is None and move_str is None:
        return jsonify({"error": "Either 'action' or 'move' parameter is required"}), 400
    
    # Convert move string to action if provided
    if action is None and move_str is not None:
        action = session.game_state.stringToAction(move_str)
        if action is None:
            return jsonify({"error": f"Invalid move string: {move_str}"}), 400
    
    # Make the move
    success = session.make_move(action)
    if not success:
        return jsonify({"error": "Invalid move"}), 400
    
    # Return updated state
    return jsonify(session.get_state())


@app.route('/api/games/<session_id>/ai_move', methods=['POST'])
def get_ai_move(session_id):
    """Get an AI move for the game."""
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Game session not found"}), 404
    
    # Check if game is over
    if session.game_state.isTerminal():
        return jsonify({
            "error": "Game is already over",
            "state": session.get_state()
        }), 400
    
    # Get AI move
    action = session.get_ai_move()
    if action is None:
        return jsonify({
            "error": "Failed to get AI move",
            "state": session.get_state()
        }), 500
    
    # Convert action to move string
    move_str = session.game_state.actionToString(action)
    
    # Return updated state with AI move
    return jsonify({
        "action": action,
        "move": move_str,
        "state": session.get_state()
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    models = session_manager.get_available_models()
    return jsonify(models)


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server to")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing models")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for move selection")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for authentication")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    global api_key, session_manager
    
    # Configure logging
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    
    # Set API key
    api_key = args.api_key
    
    # Create session manager
    session_manager = SessionManager(args.model_dir)
    
    # Start Flask server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()