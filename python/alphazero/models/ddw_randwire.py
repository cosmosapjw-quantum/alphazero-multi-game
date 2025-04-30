# python/alphazero/models/ddw_randwire.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out

class RouterModule(nn.Module):
    """Router module for dynamic wiring"""
    def __init__(self, in_channels, out_channels):
        super(RouterModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class RandWireBlock(nn.Module):
    """Random wiring block with dynamic connections"""
    def __init__(self, channels, num_nodes=32, p=0.75, seed=None):
        super(RandWireBlock, self).__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate random graph
        self.graph = self._generate_graph(num_nodes, p)
        
        # Find input and output nodes
        self.input_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        self.output_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # Ensure at least one input and output node
        if not self.input_nodes:
            self.input_nodes = [0]
        if not self.output_nodes:
            self.output_nodes = [num_nodes - 1]
        
        # Create router modules for each node
        self.routers = nn.ModuleDict()
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            if in_degree > 0:
                self.routers[str(node)] = RouterModule(in_degree * channels, channels)
        
        # Create residual blocks for each node
        self.blocks = nn.ModuleDict()
        for node in self.graph.nodes():
            self.blocks[str(node)] = ResidualBlock(channels)
        
        # Output channel router
        if len(self.output_nodes) > 1:
            self.output_router = RouterModule(len(self.output_nodes) * channels, channels)
        else:
            self.output_router = None
    
    def _generate_graph(self, num_nodes, p):
        """Generate a small-world graph with WS model"""
        k = 4  # Each node is connected to k nearest neighbors
        
        # Create small-world graph
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        
        # Convert to directed graph
        DG = nx.DiGraph()
        
        # Add nodes and edges from undirected graph
        for u, v in G.edges():
            # Ensure directed edges flow from lower to higher index
            # to avoid cycles in the network
            if u < v:
                DG.add_edge(u, v)
            else:
                DG.add_edge(v, u)
        
        return DG
    
    def forward(self, x):
        # Node outputs
        node_outputs = {}
        
        # Process input nodes
        for node in self.input_nodes:
            node_outputs[node] = self.blocks[str(node)](x)
        
        # Topological sort for processing order
        for node in nx.topological_sort(self.graph):
            # Skip input nodes
            if node in self.input_nodes:
                continue
            
            # Get inputs from predecessor nodes
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                continue
            
            # Concatenate inputs
            inputs = [node_outputs[pred] for pred in predecessors]
            if len(inputs) > 1:
                combined = torch.cat(inputs, dim=1)
                routed = self.routers[str(node)](combined)
            else:
                routed = inputs[0]
            
            # Process through residual block
            node_outputs[node] = self.blocks[str(node)](routed)
        
        # Combine outputs
        if len(self.output_nodes) > 1:
            outputs = [node_outputs[node] for node in self.output_nodes]
            combined = torch.cat(outputs, dim=1)
            return self.output_router(combined)
        else:
            return node_outputs[self.output_nodes[0]]

class DDWRandWireResNet(nn.Module):
    """Dynamic Dense-Wired Random-Wire ResNet for AlphaZero"""
    def __init__(self, input_channels, output_size, channels=128, num_blocks=20):
        super(DDWRandWireResNet, self).__init__()
        self.input_channels = input_channels
        
        # Input layer
        self.input_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # Random wire blocks
        self.rand_wire_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.rand_wire_blocks.append(
                RandWireBlock(channels, num_nodes=32, p=0.75, seed=i)
            )
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, output_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Random wire blocks
        for block in self.rand_wire_blocks:
            x = block(x)
        
        # Adaptive pooling to handle different board sizes
        # Get current size
        batch, channels, height, width = x.size()
        
        # Target size of 8x8
        target_size = min(8, height, width)
        
        # Adjust pooling target based on input dimensions
        if height != target_size or width != target_size:
            x_pooled = F.adaptive_avg_pool2d(x, (target_size, target_size))
        else:
            x_pooled = x
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x_pooled)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x_pooled)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

def test_randwire_resnet():
    # Create tensor batch (batch_size, channels, height, width)
    x = torch.randn(4, 18, 15, 15)
    
    # Create model with input channels=18, output size=225 (15x15 board)
    model = DDWRandWireResNet(input_channels=18, output_size=225)
    
    # Forward pass
    policy, value = model(x)
    
    # Check shapes
    print(f"Input shape: {x.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    
    # Check parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count:,}")

if __name__ == "__main__":
    test_randwire_resnet()