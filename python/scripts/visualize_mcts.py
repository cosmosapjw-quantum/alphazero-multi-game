#!/usr/bin/env python3
"""
Visualization script for AlphaZero MCTS search tree.

This script creates a visual representation of the Monte Carlo Tree Search for
analyzing and debugging search behavior.

Usage:
    python visualize_mcts.py [options]

Options:
    --model MODEL           Path to model file
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE             Board size (default: depends on game)
    --simulations SIMS      Number of MCTS simulations (default: 1600)
    --threads THREADS       Number of threads (default: 4)
    --depth DEPTH           Maximum depth to visualize (default: 3)
    --fen FEN               Starting position in FEN (for chess)
    --moves MOVES           Sequence of moves to make before visualization
    --output-file FILE      Output file for visualization (default: mcts_tree.png)
    --use-gpu               Use GPU for inference
    --variant               Use variant rules
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.cm import get_cmap
import torch
import _alphazero_cpp as az
from alphazero.models import DDWRandWireResNet
from alphazero.utils.visualization import plot_board

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero MCTS Visualization")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--simulations", type=int, default=1600,
                        help="Number of MCTS simulations")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--depth", type=int, default=3,
                        help="Maximum depth to visualize")
    parser.add_argument("--fen", type=str, default="",
                        help="Starting position in FEN (for chess)")
    parser.add_argument("--moves", type=str, default="",
                        help="Sequence of moves to make before visualization")
    parser.add_argument("--output-file", type=str, default="mcts_tree.png",
                        help="Output file for visualization")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for inference")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--show-all-nodes", action="store_true",
                        help="Show all nodes in the tree, not just the most visited paths")
    parser.add_argument("--max-nodes", type=int, default=50,
                        help="Maximum number of nodes to display in the visualization")
    return parser.parse_args()


def create_neural_network(model_path, game_type, board_size, use_gpu=False):
    """Create a neural network wrapper for the AI."""
    if not model_path:
        print("Using random policy (no model provided)")
        return None
    
    try:
        # Try to load with the C++ API first
        nn = az.createNeuralNetwork(model_path, game_type, board_size, use_gpu)
        print(f"Loaded model from {model_path} (C++ API)")
        return nn
    except Exception as e:
        print(f"Failed to load model with C++ API: {e}")
        
        # Try to load with PyTorch
        try:
            # Create a test game state to get input shape
            game_state = az.createGameState(game_type, board_size, False)
            tensor_rep = game_state.getEnhancedTensorRepresentation()
            input_channels = len(tensor_rep)
            action_size = game_state.getActionSpaceSize()
            
            # Create and load model
            device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            model = DDWRandWireResNet(input_channels, action_size)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Create wrapper for the PyTorch model
            class TorchNeuralNetwork(az.NeuralNetwork):
                def __init__(self, model, device):
                    super().__init__()
                    self.model = model
                    self.device = device
                
                def predict(self, state):
                    # Convert state tensor to PyTorch tensor
                    state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                    state_tensor = state_tensor.unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        policy_logits, value = self.model(state_tensor)
                        policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                        value = value.item()
                    
                    return policy, value
                
                def predictBatch(self, states, policies, values):
                    # Convert states to PyTorch tensor
                    batch_size = len(states)
                    state_tensors = []
                    
                    for i in range(batch_size):
                        state = states[i].get()
                        state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                        state_tensors.append(state_tensor)
                    
                    batch_tensor = torch.stack(state_tensors).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        policy_logits, value_tensor = self.model(batch_tensor)
                        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                        value_list = value_tensor.squeeze(-1).cpu().numpy()
                    
                    # Set output
                    for i in range(batch_size):
                        policies[i] = policy_probs[i].tolist()
                        values[i] = value_list[i]
                
                def isGpuAvailable(self):
                    return torch.cuda.is_available()
                
                def getDeviceInfo(self):
                    if torch.cuda.is_available():
                        return f"GPU: {torch.cuda.get_device_name(0)}"
                    else:
                        return "CPU"
                
                def getInferenceTimeMs(self):
                    return 0.0
                
                def getBatchSize(self):
                    return 16
                
                def getModelInfo(self):
                    return "PyTorch DDWRandWireResNet"
                
                def getModelSizeBytes(self):
                    return sum(p.numel() * 4 for p in self.model.parameters())
            
            nn = TorchNeuralNetwork(model, device)
            print(f"Loaded model from {model_path} (PyTorch)")
            return nn
        except Exception as e:
            print(f"Failed to load model with PyTorch: {e}")
            print("Using random policy network instead")
            return None


def extract_search_tree(mcts, game_state, max_depth=3, max_nodes=50, show_all_nodes=False):
    """Extract the MCTS search tree as a NetworkX graph."""
    G = nx.DiGraph()
    
    # Get root node information
    root_node = mcts.getRootNode()
    if not root_node:
        print("Error: Failed to get root node from MCTS")
        return G
    
    # Create a queue for BFS traversal
    queue = [(root_node, None, 0, "Root")]  # (node, parent_id, depth, action_str)
    
    # Track node IDs
    node_id_counter = 0
    node_ids = {}
    
    # Collect stats to determine opacity scaling
    visit_counts = []
    
    # Keep track of most visited paths
    most_visited_paths = set()
    if not show_all_nodes:
        # Find the most visited child at each level
        current = root_node
        most_visited_paths.add(current)
        
        for _ in range(max_depth):
            if not current or not current.hasChildren():
                break
            
            # Find the most visited child
            max_visits = -1
            most_visited_child = None
            
            for i in range(current.getChildCount()):
                child = current.getChild(i)
                visits = child.getVisitCount()
                
                if visits > max_visits:
                    max_visits = visits
                    most_visited_child = child
            
            if most_visited_child:
                most_visited_paths.add(most_visited_child)
                current = most_visited_child
            else:
                break
    
    # BFS traversal
    nodes_added = 0
    while queue and nodes_added < max_nodes:
        node, parent_id, depth, action_str = queue.pop(0)
        
        # Skip if we're past the max depth
        if depth > max_depth:
            continue
        
        # Skip if it's not a most-visited path and we're not showing all nodes
        if not show_all_nodes and node not in most_visited_paths and parent_id is not None:
            continue
        
        # Generate a unique ID for this node
        if node not in node_ids:
            node_ids[node] = node_id_counter
            node_id_counter += 1
        
        node_id = node_ids[node]
        
        # Get node statistics
        visits = node.getVisitCount()
        value = node.getValue()
        visit_counts.append(visits)
        
        # Add the node to the graph
        G.add_node(
            node_id,
            visits=visits,
            value=value,
            action=action_str,
            depth=depth
        )
        
        # Connect to parent
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        
        nodes_added += 1
        
        # Add children to the queue for the next level
        if depth < max_depth and node.hasChildren():
            for i in range(node.getChildCount()):
                child = node.getChild(i)
                action = node.getAction(i)
                action_str = game_state.actionToString(action)
                
                # Check if this is one of the most visited paths
                is_most_visited = (child in most_visited_paths) or show_all_nodes
                
                # Always add children of most visited paths, or add all if show_all_nodes is true
                if is_most_visited:
                    queue.append((child, node_id, depth + 1, action_str))
    
    # Calculate normalized visit counts for node sizing and color
    if visit_counts:
        max_visits = max(visit_counts)
        for node_id in G.nodes():
            visits = G.nodes[node_id]['visits']
            G.nodes[node_id]['size'] = 300 * (0.1 + 0.9 * (visits / max_visits))
            G.nodes[node_id]['color'] = visits / max_visits
    
    return G


def visualize_search_tree(G, output_file, game_state):
    """Create a visualization of the MCTS search tree."""
    if not G.nodes():
        print("Error: No nodes in the graph to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Use hierarchical layout
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Get node attributes for visualization
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Get a colormap
    cmap = plt.cm.viridis
    
    # Draw the nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        alpha=0.8
    )
    
    # Draw the edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.0,
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        edge_color='grey'
    )
    
    # Create labels with visit counts, values, and actions
    labels = {}
    for node in G.nodes():
        visits = G.nodes[node]['visits']
        value = G.nodes[node]['value']
        action = G.nodes[node]['action']
        
        # Format the label
        label = f"{action}\nV={visits}\nQ={value:.2f}"
        labels[node] = label
    
    # Draw the labels
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_family='sans-serif'
    )
    
    # Add a colorbar to show visit count scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Normalized Visit Count')
    
    # Set title
    game_name = "Unknown"
    if game_state.getGameType() == az.GameType.GOMOKU:
        game_name = "Gomoku"
    elif game_state.getGameType() == az.GameType.CHESS:
        game_name = "Chess"
    elif game_state.getGameType() == az.GameType.GO:
        game_name = "Go"
    
    plt.title(f"MCTS Search Tree for {game_name}")
    
    # Remove axis
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Search tree visualization saved to {output_file}")
    
    # Create a second figure with the current board state
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the current board state
    plot_board(game_state)
    
    # Save the board visualization
    board_file = os.path.splitext(output_file)[0] + "_board.png"
    plt.savefig(board_file, dpi=300, bbox_inches='tight')
    print(f"Board visualization saved to {board_file}")


def main(args):
    """Main function to visualize the MCTS search tree."""
    # Convert game type string to enum
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Default board sizes
    if args.size <= 0:
        if args.game == "gomoku":
            board_size = 15
        elif args.game == "chess":
            board_size = 8  # Chess is always 8x8
        elif args.game == "go":
            board_size = 9  # Use smaller board for visualization
        else:
            board_size = 15
    else:
        board_size = args.size
    
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Simulations: {args.simulations}")
    print(f"Max depth: {args.depth}")
    
    # Create the game state
    game_state = az.createGameState(game_type, board_size, args.variant)
    
    # Set up a specific chess position if FEN is provided
    if args.fen and game_type == az.GameType.CHESS:
        if hasattr(game_state, 'setFromFEN'):
            game_state.setFromFEN(args.fen)
            print(f"Set chess position from FEN: {args.fen}")
        else:
            print("Warning: setFromFEN not available for this chess implementation")
    
    # Apply sequence of moves if provided
    if args.moves:
        moves = args.moves.split()
        for move_str in moves:
            action = game_state.stringToAction(move_str)
            if action is not None:
                game_state.makeMove(action)
                print(f"Applied move: {move_str}")
            else:
                print(f"Warning: Invalid move format: {move_str}")
    
    # Create neural network
    nn = create_neural_network(args.model, game_type, board_size, args.use_gpu)
    
    # Create transposition table
    tt = az.TranspositionTable(1048576, 1024)
    
    # Create MCTS
    mcts = az.ParallelMCTS(
        game_state, nn, tt,
        args.threads, args.simulations
    )
    
    # Set MCTS parameters
    mcts.setCPuct(1.5)
    mcts.setFpuReduction(0.0)
    
    # Set progress callback
    def progress_callback(current, total):
        print(f"\rRunning MCTS search: {current}/{total} simulations ({current/total*100:.1f}%)", end="")
    
    mcts.setProgressCallback(progress_callback)
    
    print("Running MCTS search...")
    mcts.search()
    print("\nSearch complete")
    
    # Extract and visualize the search tree
    G = extract_search_tree(mcts, game_state, args.depth, args.max_nodes, args.show_all_nodes)
    visualize_search_tree(G, args.output_file, game_state)


if __name__ == "__main__":
    args = parse_args()
    main(args)