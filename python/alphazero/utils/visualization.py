"""
Visualization utilities for AlphaZero.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image
import _alphazero_cpp as az


def plot_board(game_state, last_move=None, policy=None, figsize=(8, 8), dpi=100):
    """
    Plot the current board state.
    
    Args:
        game_state (IGameState): Game state to visualize
        last_move (int, optional): Index of the last move to highlight
        policy (list, optional): Policy vector to visualize as a heatmap
        figsize (tuple): Figure size
        dpi (int): DPI for rendering
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    game_type = game_state.getGameType()
    board_size = game_state.getBoardSize()
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if game_type == az.GameType.GOMOKU:
        return _plot_gomoku_board(ax, game_state, last_move, policy)
    elif game_type == az.GameType.CHESS:
        return _plot_chess_board(ax, game_state, last_move, policy)
    elif game_type == az.GameType.GO:
        return _plot_go_board(ax, game_state, last_move, policy)
    else:
        ax.text(0.5, 0.5, "Unsupported game type", 
                ha='center', va='center', fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def _plot_gomoku_board(ax, game_state, last_move=None, policy=None):
    """
    Plot a Gomoku board.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        game_state (GomokuState): Gomoku game state
        last_move (int, optional): Index of the last move to highlight
        policy (list, optional): Policy vector to visualize as a heatmap
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    board_size = game_state.getBoardSize()
    
    # Draw background
    ax.set_facecolor('#E8BB77')
    
    # Draw grid lines
    for i in range(board_size):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Set limits and aspect ratio
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(-0.5, board_size - 0.5)
    ax.set_aspect('equal')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw grid coordinates
    for i in range(board_size):
        ax.text(i, -0.7, chr(65 + i), ha='center', va='center', fontsize=10)
        ax.text(-0.7, i, str(board_size - i), ha='center', va='center', fontsize=10)
    
    # Draw policy heatmap if provided
    if policy is not None and len(policy) >= board_size * board_size:
        policy_2d = np.array(policy[:board_size * board_size]).reshape(board_size, board_size)
        cmap = LinearSegmentedColormap.from_list('policy', 
                                               [(1, 1, 1, 0), (0, 0.7, 0, 0.7)])
        ax.imshow(policy_2d, origin='upper', extent=(-0.5, board_size - 0.5, 
                                                   -0.5, board_size - 0.5),
                 cmap=cmap, vmin=0, vmax=max(0.01, np.max(policy_2d)))
    
    # Draw stones
    for y in range(board_size):
        for x in range(board_size):
            action = y * board_size + x
            stone = 0
            
            # Check if this position is occupied
            if hasattr(game_state, 'is_occupied'):
                if game_state.is_occupied(action):
                    # Get stone color (1=black, 2=white)
                    stone = game_state.get_board()[y][x]
            
            if stone == 1:  # Black stone
                circle = plt.Circle((x, y), 0.4, color='black')
                ax.add_patch(circle)
            elif stone == 2:  # White stone
                circle = plt.Circle((x, y), 0.4, color='white', edgecolor='black')
                ax.add_patch(circle)
    
    # Highlight last move
    if last_move is not None and last_move >= 0 and last_move < board_size * board_size:
        y, x = divmod(last_move, board_size)
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # Set title
    if game_state.isTerminal():
        result = game_state.getGameResult()
        if result == az.GameResult.WIN_PLAYER1:
            ax.set_title("Game over: Black wins")
        elif result == az.GameResult.WIN_PLAYER2:
            ax.set_title("Game over: White wins")
        elif result == az.GameResult.DRAW:
            ax.set_title("Game over: Draw")
    else:
        player = game_state.getCurrentPlayer()
        ax.set_title(f"Current player: {'Black' if player == 1 else 'White'}")
    
    return ax.figure


def _plot_chess_board(ax, game_state, last_move=None, policy=None):
    """
    Plot a Chess board.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        game_state (ChessState): Chess game state
        last_move (int, optional): Index of the last move to highlight
        policy (list, optional): Policy vector to visualize as a heatmap
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    board_size = 8  # Chess is always 8x8
    
    # Draw squares
    for y in range(board_size):
        for x in range(board_size):
            color = '#FECE9E' if (x + y) % 2 == 0 else '#D18B47'
            rect = patches.Rectangle((x, 7-y), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)
    
    # Set limits and aspect ratio
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect('equal')
    
    # Draw coordinates
    for i in range(board_size):
        ax.text(i + 0.5, -0.3, chr(97 + i), ha='center', va='center', fontsize=10)
        ax.text(-0.3, i + 0.5, str(8 - i), ha='center', va='center', fontsize=10)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw policy heatmap if provided
    if policy is not None and len(policy) >= 64 * 73:  # 64 squares, max 73 moves per square
        # Convert policy to 8x8 board for visualization
        # For chess, we sum up all possible moves from each square
        policy_2d = np.zeros((8, 8))
        for src in range(64):
            src_y, src_x = divmod(src, 8)
            src_y = 7 - src_y  # Flip y-coordinate for display
            policy_2d[src_y, src_x] = sum(policy[src * 73:(src + 1) * 73])
        
        # Normalize and draw heatmap
        if np.max(policy_2d) > 0:
            policy_2d /= np.max(policy_2d)
            cmap = LinearSegmentedColormap.from_list('policy', 
                                                  [(1, 1, 1, 0), (0, 0.7, 0, 0.7)])
            ax.imshow(policy_2d, origin='upper', extent=(0, 8, 0, 8),
                    cmap=cmap, alpha=0.7, vmin=0, vmax=1)
    
    # Draw pieces based on FEN representation (if available)
    if hasattr(game_state, 'get_fen'):
        fen = game_state.get_fen()
        if fen:
            pieces = fen.split(' ')[0]
            row, col = 0, 0
            
            piece_symbols = {
                'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
            }
            
            for char in pieces:
                if char == '/':
                    row += 1
                    col = 0
                elif char.isdigit():
                    col += int(char)
                elif char in piece_symbols:
                    color = 'white' if char.isupper() else 'black'
                    text = ax.text(col + 0.5, 7.5 - row, piece_symbols[char], 
                                 color=color, ha='center', va='center', 
                                 fontsize=32, fontweight='bold')
                    text.set_path_effects([
                        plt.patheffects.withStroke(linewidth=2, foreground='black')
                    ])
                    col += 1
    
    # Set title
    if game_state.isTerminal():
        result = game_state.getGameResult()
        if result == az.GameResult.WIN_PLAYER1:
            ax.set_title("Game over: White wins")
        elif result == az.GameResult.WIN_PLAYER2:
            ax.set_title("Game over: Black wins")
        elif result == az.GameResult.DRAW:
            ax.set_title("Game over: Draw")
    else:
        player = game_state.getCurrentPlayer()
        ax.set_title(f"Current player: {'White' if player == 1 else 'Black'}")
    
    return ax.figure


def _plot_go_board(ax, game_state, last_move=None, policy=None):
    """
    Plot a Go board.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        game_state (GoState): Go game state
        last_move (int, optional): Index of the last move to highlight
        policy (list, optional): Policy vector to visualize as a heatmap
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    board_size = game_state.getBoardSize()
    
    # Draw background
    ax.set_facecolor('#DEB887')
    
    # Draw grid lines
    for i in range(board_size):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Set limits and aspect ratio
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(-0.5, board_size - 0.5)
    ax.set_aspect('equal')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw grid coordinates
    for i in range(board_size):
        ax.text(i, -0.7, chr(65 + i), ha='center', va='center', fontsize=10)
        ax.text(-0.7, i, str(board_size - i), ha='center', va='center', fontsize=10)
    
    # Draw star points (for 9x9 or 19x19 boards)
    if board_size == 9:
        star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for x, y in star_points:
            ax.plot(x, y, 'ko', markersize=5)
    elif board_size == 19:
        star_points = [(3, 3), (3, 9), (3, 15), 
                      (9, 3), (9, 9), (9, 15), 
                      (15, 3), (15, 9), (15, 15)]
        for x, y in star_points:
            ax.plot(x, y, 'ko', markersize=5)
    
    # Draw policy heatmap if provided
    if policy is not None and len(policy) >= board_size * board_size + 1:  # +1 for pass move
        policy_2d = np.array(policy[:board_size * board_size]).reshape(board_size, board_size)
        cmap = LinearSegmentedColormap.from_list('policy', 
                                               [(1, 1, 1, 0), (0, 0.7, 0, 0.7)])
        ax.imshow(policy_2d, origin='upper', extent=(-0.5, board_size - 0.5, 
                                                   -0.5, board_size - 0.5),
                 cmap=cmap, vmin=0, vmax=max(0.01, np.max(policy_2d)))
    
    # Draw stones
    if hasattr(game_state, 'get_board'):
        board = game_state.get_board()
        for y in range(board_size):
            for x in range(board_size):
                stone = board[y][x]
                if stone == 1:  # Black stone
                    circle = plt.Circle((x, y), 0.4, color='black')
                    ax.add_patch(circle)
                elif stone == 2:  # White stone
                    circle = plt.Circle((x, y), 0.4, color='white', edgecolor='black')
                    ax.add_patch(circle)
    
    # Highlight last move
    if last_move is not None and last_move >= 0 and last_move < board_size * board_size:
        y, x = divmod(last_move, board_size)
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # Set title
    if game_state.isTerminal():
        result = game_state.getGameResult()
        if result == az.GameResult.WIN_PLAYER1:
            ax.set_title("Game over: Black wins")
        elif result == az.GameResult.WIN_PLAYER2:
            ax.set_title("Game over: White wins")
        elif result == az.GameResult.DRAW:
            ax.set_title("Game over: Draw")
    else:
        player = game_state.getCurrentPlayer()
        ax.set_title(f"Current player: {'Black' if player == 1 else 'White'}")
    
    return ax.figure


def plot_training_history(history, figsize=(15, 5)):
    """
    Plot training history metrics.
    
    Args:
        history (dict): Dictionary containing training history metrics
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot total loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot policy loss
    ax = axes[1]
    if 'train_policy_loss' in history:
        ax.plot(history['train_policy_loss'], label='Train')
    if 'val_policy_loss' in history:
        ax.plot(history['val_policy_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot value loss
    ax = axes[2]
    if 'train_value_loss' in history:
        ax.plot(history['train_value_loss'], label='Train')
    if 'val_value_loss' in history:
        ax.plot(history['val_value_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Value Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_elo_history(elo_tracker, player_ids=None, figsize=(10, 6)):
    """
    Plot ELO rating history for players.
    
    Args:
        elo_tracker (EloRating): ELO rating tracker
        player_ids (list, optional): List of player IDs to plot. If None, plot all.
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all player IDs if not specified
    if player_ids is None:
        player_ids = list(elo_tracker.ratings.keys())
    
    # Plot each player's rating history
    for player_id in player_ids:
        history = elo_tracker.get_history(player_id)
        if not history:
            continue
        
        # Extract timestamps and ratings
        timestamps = [datetime.datetime.fromisoformat(h[0]) for h in history]
        ratings = [h[1] for h in history]
        
        # Plot the ratings
        ax.plot(timestamps, ratings, 'o-', label=player_id)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('ELO Rating')
    ax.set_title('ELO Rating History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis date labels
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_search_tree(mcts, max_depth=2, figsize=(12, 8)):
    """
    Plot the MCTS search tree.
    
    Args:
        mcts (ParallelMCTS): MCTS instance
        max_depth (int): Maximum depth to visualize
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # This is a placeholder - implementing a full search tree visualization
    # would require more complex graph drawing logic
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get search information as text
    search_info = mcts.getSearchInfo()
    
    # Display as text for now
    ax.text(0.5, 0.5, search_info, ha='center', va='center', 
           fontfamily='monospace', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig