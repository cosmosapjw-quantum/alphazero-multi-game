#!/usr/bin/env python3
"""
Simplified model export utility for AlphaZero models.
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
import time

class SimplifiedModel(nn.Module):
    """A simplified model for demonstration purposes."""
    def __init__(self, input_channels, action_size, channels=16, blocks=2):
        super().__init__()
        self.input_channels = input_channels
        
        # Input layer
        self.input_conv = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(blocks):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels)
            )
            self.res_blocks.append(block)
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Input layer
        x = torch.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)
        
        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

def export_model(input_channels, action_size, output_dir="exported_models", 
                channels=16, blocks=2, device="cpu"):
    """Create a random model and export to TorchScript."""
    print(f"Creating model with {channels} channels and {blocks} blocks")
    
    # Create the model
    model = SimplifiedModel(input_channels, action_size, channels, blocks)
    print("Model created successfully")
    
    # Move to device
    print(f"Moving model to device: {device}")
    model.to(device)
    model.eval()
    
    # Get model filename
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"simple_model_channels{channels}_blocks{blocks}.pt")
    
    # Save the model state dict
    print(f"Saving model state dict to {model_path}")
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully")
    
    # Export to TorchScript
    print("Exporting to TorchScript")
    example_input = torch.randn(1, input_channels, 8, 8).to(device)
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    ts_path = os.path.join(output_dir, f"simple_model_channels{channels}_blocks{blocks}_torchscript.pt")
    print(f"Saving TorchScript model to {ts_path}")
    traced_model.save(ts_path)
    print(f"Exported TorchScript model to {ts_path}")
    
    return model, ts_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplified Model Export Utility")
    parser.add_argument("--channels", type=int, default=16,
                        help="Number of model channels (default: 16)")
    parser.add_argument("--blocks", type=int, default=2,
                        help="Number of residual blocks (default: 2)")
    parser.add_argument("--output-dir", type=str, default="exported_models",
                        help="Output directory (default: exported_models)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (default: auto-detect)")
    args = parser.parse_args()
    
    # Get board size and action size for Gomoku (hardcoded for simplicity)
    board_size = 15
    input_channels = 11  # Typically used for Gomoku in AlphaZero
    action_size = board_size * board_size
    
    # Determine device
    device = args.device
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Export the model
    start_time = time.time()
    model, ts_path = export_model(
        input_channels, 
        action_size, 
        args.output_dir,
        args.channels,
        args.blocks,
        device
    )
    end_time = time.time()
    
    print(f"Model creation and export completed in {end_time - start_time:.2f} seconds")
    
    # Test the exported model
    print("\nTesting exported model...")
    example_input = torch.randn(1, input_channels, 8, 8).to(device)
    
    # Get predictions from original model
    with torch.no_grad():
        original_policy, original_value = model(example_input)
    
    # Load and test the exported model
    exported_model = torch.jit.load(ts_path)
    exported_model.eval()
    
    with torch.no_grad():
        exported_policy, exported_value = exported_model(example_input)
    
    # Check if outputs match
    policy_match = torch.allclose(original_policy, exported_policy)
    value_match = torch.allclose(original_value, exported_value)
    
    print(f"Policy outputs match: {policy_match}")
    print(f"Value outputs match: {value_match}")
    
    print("\nModel export completed successfully!")

if __name__ == "__main__":
    main() 