#!/usr/bin/env python3
"""
AlphaZero DDW-RandWire-ResNet Model Architecture

This script defines the DDW-RandWire-ResNet architecture for AlphaZero
and provides utilities to create, save, and export models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
import math
import argparse
import os
from typing import List, Dict, Tuple, Optional

class RouterModule(nn.Module):
    """SE-style router module for dynamic wiring"""
    def __init__(self, channels: int, reduction: int = 16):
        super(RouterModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # Excite
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Apply scaling
        return x * y.view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    """Residual block for each node in the graph"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class GraphNode(nn.Module):
    """Node in the DDW-RandWire graph"""
    def __init__(self, in_channels: int, out_channels: int):
        super(GraphNode, self).__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.router = RouterModule(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.router(x)
        return x

def generate_random_graph(num_nodes: int, p: float = 0.75, seed: Optional[int] = None) -> nx.DiGraph:
    """Generate a random directed graph"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Create a graph with nodes but no edges
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    
    # Add input and output node
    input_node = 0
    output_node = num_nodes - 1
    
    # Ensure each node (except output) has at least one outgoing edge
    for node in range(num_nodes - 1):
        # Choose a random target node that comes after the current node
        target = random.randint(node + 1, num_nodes - 1)
        G.add_edge(node, target)
    
    # Add random edges with probability p
    for node in range(num_nodes - 1):
        for target in range(node + 1, num_nodes):
            if random.random() < p and not G.has_edge(node, target):
                G.add_edge(node, target)
    
    # Ensure there's a path from input to output
    if not nx.has_path(G, input_node, output_node):
        G.add_edge(input_node, output_node)
    
    return G

class DDWRandWireResNet(nn.Module):
    """DDW-RandWire-ResNet architecture for AlphaZero"""
    def __init__(self, input_channels: int, action_space_size: int, board_size: int,
                 num_nodes: int = 32, p: float = 0.75, node_channels: int = 128,
                 seed: Optional[int] = None):
        super(DDWRandWireResNet, self).__init__()
        
        self.input_channels = input_channels
        self.action_space_size = action_space_size
        self.board_size = board_size
        
        # Stem layer to process input
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, node_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(node_channels),
            nn.ReLU()
        )
        
        # Generate random graph
        self.graph = generate_random_graph(num_nodes, p, seed)
        self.input_node = 0
        self.output_node = num_nodes - 1
        
        # Create nodes
        self.nodes = nn.ModuleList([
            GraphNode(node_channels, node_channels) for _ in range(num_nodes)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(node_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, action_space_size),
            nn.Softmax(dim=1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(node_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process input through stem
        x = self.stem(x)
        
        # Store node outputs
        node_outputs = {self.input_node: x}
        
        # Process nodes in topological order
        for node in nx.topological_sort(self.graph):
            if node == self.input_node:
                continue
            
            # Get all inputs to this node
            inputs = [node_outputs[pred] for pred in self.graph.predecessors(node)]
            
            if not inputs:
                continue
            
            # Average inputs if multiple
            if len(inputs) > 1:
                node_input = torch.mean(torch.stack(inputs), dim=0)
            else:
                node_input = inputs[0]
            
            # Process through node
            node_output = self.nodes[node](node_input)
            node_outputs[node] = node_output
        
        # Get final output from the output node
        output = node_outputs[self.output_node]
        
        # Process through policy and value heads
        policy = self.policy_head(output)
        value = self.value_head(output)
        
        return policy, value
    
    def get_graph_info(self) -> Dict:
        """Get information about the graph topology"""
        return {
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes),
            "graph_dict": {
                "nodes": list(self.graph.nodes),
                "edges": list(self.graph.edges)
            }
        }

# Example usage
def create_gomoku_model(board_size: int = 15, seed: Optional[int] = None) -> DDWRandWireResNet:
    """Create a model for Gomoku"""
    input_channels = 8  # Current player stones, opponent stones, history, and auxiliary channels
    action_space_size = board_size * board_size
    
    return DDWRandWireResNet(
        input_channels=input_channels,
        action_space_size=action_space_size,
        board_size=board_size,
        seed=seed
    )

def export_model(model: nn.Module, path: str):
    """Export model to TorchScript format"""
    # Create example input
    example_input = torch.randn(1, model.input_channels, model.board_size, model.board_size)
    
    # Trace the model
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the model
    traced_model.save(path)
    print(f"Model exported to {path}")

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Model Creation Tool")
    parser.add_argument("--game", choices=["gomoku", "chess", "go"], default="gomoku",
                       help="Game type")
    parser.add_argument("--board-size", type=int, default=15,
                       help="Board size")
    parser.add_argument("--output", type=str, default="models/gomoku_model.pt",
                       help="Output model path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for model initialization")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create model based on game type
    if args.game == "gomoku":
        model = create_gomoku_model(args.board_size, args.seed)
    else:
        raise ValueError(f"Game type {args.game} not implemented yet")
    
    # Print model info
    print(f"Created model for {args.game.capitalize()} (board size: {args.board_size})")
    print(f"Graph info: {model.get_graph_info()}")
    
    # Export model
    export_model(model, args.output)

if __name__ == "__main__":
    main()