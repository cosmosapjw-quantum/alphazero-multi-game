#!/usr/bin/env python3
"""
Interactive Chess game against AlphaZero AI.

Usage:
    python play_chess.py [options]

Options:
    --model MODEL       Path to model file
    --simulations SIMS  Number of MCTS simulations (default: 800)
    --threads THREADS   Number of threads (default: 4)
    --chess960          Use Chess960 rules
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import _alphazero_cpp as az
from alphazero.models import DDWRandWireResNet
from alphazero.utils.visualization import plot_board

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ChessGUI:
    """
    GUI for playing Chess against AlphaZero.
    """
    
    def __init__(self, master, args):
        self.master = master
        master.title("AlphaZero Chess")
        
        # Parse arguments
        self.simulations = args.simulations
        self.threads = args.threads
        self.chess960 = args.chess960
        self.model_path = args.model
        
        # State variables
        self.selected_square = None
        
        # Set up game and AI
        self.setup_game()
        
        # Create widgets
        self.create_widgets()
        
        # Draw the board
        self.update_board()
    
    def setup_game(self):
        """Set up the game state and AI."""
        # Create game state
        self.game_state = az.createGameState(
            az.GameType.CHESS, 0, self.chess960
        )
        
        # Load model if provided
        if self.model_path:
            try:
                self.network = az.createNeuralNetwork(
                    self.model_path, az.GameType.CHESS, 0
                )
                self.status_text.set(f"Loaded model: {self.model_path}")
            except Exception as e:
                self.network = None
                print(f"Error loading model: {e}")
                self.status_text.set("Using random policy (no model loaded)")
        else:
            self.network = None
            self.status_text.set("Using random policy (no model provided)")
        
        # Create transposition table
        self.tt = az.TranspositionTable(1048576, 1024)
        
        # Create MCTS
        self.mcts = az.ParallelMCTS(
            self.game_state, self.network, self.tt, 
            self.threads, self.simulations
        )
        
        # Set MCTS parameters
        self.mcts.setCPuct(1.5)
        self.mcts.setFpuReduction(0.0)
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Frame for the board
        self.board_frame = tk.Frame(self.master)
        self.board_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create a figure for the board
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Add the figure to the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.board_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Status text
        self.status_text = tk.StringVar()
        self.status_text.set("Your turn (White)")
        self.status_label = tk.Label(self.master, textvariable=self.status_text, 
                                    bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control frame
        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # New game button
        self.new_game_button = tk.Button(self.control_frame, text="New Game", 
                                        command=self.new_game)
        self.new_game_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Undo button
        self.undo_button = tk.Button(self.control_frame, text="Undo", 
                                    command=self.undo_move)
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # AI move button
        self.ai_move_button = tk.Button(self.control_frame, text="AI Move", 
                                       command=self.ai_move)
        self.ai_move_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Quit button
        self.quit_button = tk.Button(self.control_frame, text="Quit", 
                                    command=self.master.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def update_board(self):
        """Update the board display."""
        self.ax.clear()
        
        # Get the last move if available
        history = self.game_state.getMoveHistory()
        last_move = history[-1] if history else None
        
        # Get policy from MCTS if available
        policy = None
        if self.mcts and not self.game_state.isTerminal():
            # Run a quick search to get policy
            self.mcts.setNumSimulations(100)
            self.mcts.search()
            policy = self.mcts.getActionProbabilities(0.1)
            self.mcts.setNumSimulations(self.simulations)
        
        # Plot the board
        plot_board(self.game_state, last_move, policy, figsize=(8, 8))
        
        # Highlight selected square if any
        if self.selected_square is not None:
            x, y = self.selected_square
            rect = plt.Rectangle((x, y), 1, 1, linewidth=3, 
                               edgecolor='blue', facecolor='none')
            self.ax.add_patch(rect)
        
        # Update canvas
        self.canvas.draw()
        
        # Update status text
        if self.game_state.isTerminal():
            result = self.game_state.getGameResult()
            if result == az.GameResult.WIN_PLAYER1:
                self.status_text.set("Game over: White wins!")
            elif result == az.GameResult.WIN_PLAYER2:
                self.status_text.set("Game over: Black wins!")
            elif result == az.GameResult.DRAW:
                self.status_text.set("Game over: Draw!")
        else:
            player = self.game_state.getCurrentPlayer()
            self.status_text.set(f"Current player: {'White' if player == 1 else 'Black'}")
            
            # Additional info for selected square
            if self.selected_square is not None:
                x, y = self.selected_square
                square_idx = (7 - y) * 8 + x
                self.status_text.set(f"{self.status_text.get()} - Selected: {chr(97 + x)}{8 - y}")
    
    def on_click(self, event):
        """Handle mouse clicks on the board."""
        # Ignore clicks outside the board or if game is over
        if (event.xdata is None or event.ydata is None or 
                self.game_state.isTerminal()):
            return
        
        # Convert to board coordinates
        x = int(event.xdata)
        y = int(event.ydata)
        
        # Check if within board boundaries
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return
        
        if self.selected_square is None:
            # First click - select a square
            self.selected_square = (x, y)
            self.update_board()
        else:
            # Second click - try to make a move
            src_x, src_y = self.selected_square
            src_idx = (7 - src_y) * 8 + src_x
            dst_idx = (7 - y) * 8 + x
            
            # Convert to chess action (source * 73 + offset)
            # For simplicity, just try all possible actions from the source square
            moved = False
            for offset in range(73):  # Max possible moves per square is 73
                action = src_idx * 73 + offset
                if action < self.game_state.getActionSpaceSize() and self.game_state.isLegalMove(action):
                    # Check if this is the move we want
                    temp_state = self.game_state.clone()
                    temp_state.makeMove(action)
                    
                    # Get the destination square from the move string
                    move_str = self.game_state.actionToString(action)
                    if len(move_str) >= 4:
                        dst_file = ord(move_str[2]) - ord('a')
                        dst_rank = 8 - int(move_str[3])
                        if dst_file == x and dst_rank == y:
                            # This is the move we want
                            self.game_state.makeMove(action)
                            self.mcts.updateWithMove(action)
                            moved = True
                            break
            
            # Reset selection
            self.selected_square = None
            
            if moved:
                # Update the board
                self.update_board()
                
                # Make AI move if game is not over
                if not self.game_state.isTerminal():
                    self.master.after(100, self.ai_move)
            else:
                # Invalid move, just update to clear selection
                self.update_board()
    
    def ai_move(self):
        """Make an AI move."""
        if self.game_state.isTerminal():
            return
        
        # Set status
        self.status_text.set("AI is thinking...")
        self.master.update()
        
        # Run search
        self.mcts.search()
        
        # Select action
        action = self.mcts.selectAction(False, 0.0)
        
        # Make move
        self.game_state.makeMove(action)
        
        # Update MCTS tree
        self.mcts.updateWithMove(action)
        
        # Update the board
        self.update_board()
    
    def new_game(self):
        """Start a new game."""
        # Create new game state
        self.game_state = az.createGameState(
            az.GameType.CHESS, 0, self.chess960
        )
        
        # Create new MCTS
        self.mcts = az.ParallelMCTS(
            self.game_state, self.network, self.tt, 
            self.threads, self.simulations
        )
        
        # Set MCTS parameters
        self.mcts.setCPuct(1.5)
        self.mcts.setFpuReduction(0.0)
        
        # Reset selection
        self.selected_square = None
        
        # Update the board
        self.update_board()
    
    def undo_move(self):
        """Undo the last two moves (player and AI)."""
        # Undo twice to get back to player's turn
        for _ in range(2):
            if self.game_state.undoMove():
                # MCTS tree cannot be easily updated for undo, so recreate it
                self.mcts = az.ParallelMCTS(
                    self.game_state, self.network, self.tt, 
                    self.threads, self.simulations
                )
                self.mcts.setCPuct(1.5)
                self.mcts.setFpuReduction(0.0)
        
        # Reset selection
        self.selected_square = None
        
        # Update the board
        self.update_board()


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Chess")
    parser.add_argument("--model", type=str, default="",
                        help="Path to the model file")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--chess960", action="store_true",
                        help="Use Chess960 rules")
    return parser.parse_args()


def main():
    args = parse_args()
    
    root = tk.Tk()
    app = ChessGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()