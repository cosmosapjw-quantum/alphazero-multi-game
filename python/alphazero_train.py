#!/usr/bin/env python3
"""
AlphaZero Neural Network Training Script

This script trains a neural network for the AlphaZero AI engine using self-play data.
It supports training for Gomoku, Chess, and Go.
"""

import os
import sys
import json
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import datetime

# Import pyalphazero if available (for data processing using C++ code)
try:
    import pyalphazero
    HAS_PYALPHAZERO = True
except ImportError:
    HAS_PYALPHAZERO = False
    print("Warning: pyalphazero module not found, using pure Python implementation")

class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training examples"""
    
    def __init__(self, data_dir, game_type):
        """
        Initialize dataset from game records
        
        Args:
            data_dir: Directory containing game record JSON files
            game_type: Type of game ('gomoku', 'chess', 'go')
        """
        self.game_type = game_type
        self.examples = []
        
        # Load data using C++ if available
        if HAS_PYALPHAZERO:
            self._load_data_cpp(data_dir)
        else:
            self._load_data_python(data_dir)
            
        print(f"Loaded {len(self.examples)} training examples")
    
    def _load_data_cpp(self, data_dir):
        """Load data using C++ bindings"""
        # Create dataset
        dataset = pyalphazero.Dataset()
        
        # Map game type string to enum
        game_type_map = {
            'gomoku': pyalphazero.GameType.GOMOKU,
            'chess': pyalphazero.GameType.CHESS,
            'go': pyalphazero.GameType.GO
        }
        game_type_enum = game_type_map.get(self.game_type.lower())
        
        if not game_type_enum:
            raise ValueError(f"Unknown game type: {self.game_type}")
        
        # Load all game records in the directory
        files = glob.glob(os.path.join(data_dir, '*.json'))
        for file in tqdm(files, desc="Loading games"):
            try:
                record = pyalphazero.GameRecord.loadFromFile(file)
                dataset.addGameRecord(record, True)  # Use enhanced features
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Extract examples with augmentations
        dataset.extractExamples(True)
        
        # Get all examples
        states_batch, policies_batch, values_batch = dataset.getBatch(dataset.size())
        
        # Convert to PyTorch tensors and add to examples
        for i in range(len(values_batch)):
            state = torch.FloatTensor(states_batch[i])
            policy = torch.FloatTensor(policies_batch[i])
            value = torch.FloatTensor([values_batch[i]])
            self.examples.append((state, policy, value))
    
    def _load_data_python(self, data_dir):
        """Load data using pure Python implementation"""
        # Load all game records in the directory
        files = glob.glob(os.path.join(data_dir, '*.json'))
        for file in tqdm(files, desc="Loading games"):
            try:
                with open(file, 'r') as f:
                    game_record = json.load(f)
                
                # Process moves
                for move in game_record.get('moves', []):
                    # Simple tensor representation (this is a placeholder)
                    # In a real implementation, we would reconstruct the game state
                    # For now, we'll create a random tensor
                    channels = 8  # Basic channels for Gomoku
                    board_size = game_record.get('board_size', 15)
                    
                    # Create a random state tensor (placeholder)
                    state = torch.rand(channels, board_size, board_size)
                    
                    # Get policy and value
                    policy = torch.FloatTensor(move.get('policy', []))
                    value = torch.FloatTensor([move.get('value', 0.0)])
                    
                    # Add to examples
                    if len(policy) > 0:
                        self.examples.append((state, policy, value))
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class AlphaZeroNetwork(nn.Module):
    """AlphaZero neural network with policy and value heads"""
    
    def __init__(self, game_type, board_size, num_channels=128, num_res_blocks=10):
        """
        Initialize network
        
        Args:
            game_type: Type of game ('gomoku', 'chess', 'go')
            board_size: Size of the board
            num_channels: Number of channels in convolutional layers
            num_res_blocks: Number of residual blocks
        """
        super(AlphaZeroNetwork, self).__init__()
        
        self.game_type = game_type
        self.board_size = board_size
        
        # Determine input channels based on game type
        if game_type.lower() == 'gomoku':
            self.input_channels = 8  # Current player, opponent, history planes, auxiliary
            self.action_size = board_size * board_size
        elif game_type.lower() == 'chess':
            self.input_channels = 14  # 6 pieces * 2 colors + auxiliary
            self.action_size = 64 * 73  # 64 squares, up to 73 moves per square
        elif game_type.lower() == 'go':
            self.input_channels = 8  # Current player, opponent, history planes, auxiliary
            self.action_size = board_size * board_size + 1  # +1 for pass move
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
        
        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._build_res_block(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, self.action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def _build_res_block(self, num_channels):
        """Build a residual block"""
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
    
    def forward(self, x):
        """Forward pass"""
        # Initial block
        x = self.conv_block(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x += residual
            x = F.relu(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

def train(model, train_loader, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    # Training loop
    progress = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (states, policies, values) in enumerate(progress):
        # Move data to device
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        policy_logits, value_preds = model(states)
        
        # Calculate loss
        policy_loss = F.cross_entropy(policy_logits, policies)
        value_loss = F.mse_loss(value_preds, values)
        loss = policy_loss + value_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        
        # Update progress bar
        progress.set_postfix({
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        })
    
    # Log metrics
    avg_loss = total_loss / len(train_loader)
    avg_policy_loss = policy_loss_sum / len(train_loader)
    avg_value_loss = value_loss_sum / len(train_loader)
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/policy', avg_policy_loss, epoch)
    writer.add_scalar('Loss/value', avg_value_loss, epoch)
    
    return avg_loss

def validate(model, val_loader, device, epoch, writer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    with torch.no_grad():
        for states, policies, values in tqdm(val_loader, desc="Validation"):
            # Move data to device
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Forward pass
            policy_logits, value_preds = model(states)
            
            # Calculate loss
            policy_loss = F.cross_entropy(policy_logits, policies)
            value_loss = F.mse_loss(value_preds, values)
            loss = policy_loss + value_loss
            
            # Update metrics
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
    
    # Log metrics
    avg_loss = total_loss / len(val_loader)
    avg_policy_loss = policy_loss_sum / len(val_loader)
    avg_value_loss = value_loss_sum / len(val_loader)
    
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    writer.add_scalar('Loss/val_policy', avg_policy_loss, epoch)
    writer.add_scalar('Loss/val_value', avg_value_loss, epoch)
    
    return avg_loss

def save_model(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def export_to_torchscript(model, example_input, filename):
    """Export model to TorchScript format for C++ loading"""
    model.eval()
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(filename)
    print(f"Model exported to {filename}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="AlphaZero Neural Network Training")
    parser.add_argument("--data-dir", required=True, help="Directory containing game records")
    parser.add_argument("--output-dir", default="models", help="Directory to save models")
    parser.add_argument("--game", default="gomoku", choices=["gomoku", "chess", "go"], help="Game type")
    parser.add_argument("--board-size", type=int, default=15, help="Board size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels in network")
    parser.add_argument("--res-blocks", type=int, default=10, help="Number of residual blocks")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume from")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.output_dir, f"logs_{args.game}_{timestamp}")
    writer = SummaryWriter(log_dir)
    
    # Load dataset
    dataset = AlphaZeroDataset(args.data_dir, args.game)
    
    # Split dataset into training and validation
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = AlphaZeroNetwork(
        args.game,
        args.board_size,
        num_channels=args.channels,
        num_res_blocks=args.res_blocks
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Load checkpoint if provided
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"{args.game}_checkpoint_{epoch}.pt")
        save_model(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, f"{args.game}_best.pt")
            save_model(model, optimizer, epoch, val_loss, best_model_path)
            print(f"New best model saved (val_loss: {val_loss:.4f})")
    
    # Export final model to TorchScript
    dummy_input = torch.zeros(1, model.input_channels, args.board_size, args.board_size).to(device)
    export_path = os.path.join(args.output_dir, f"{args.game}_final.pt")
    export_to_torchscript(model, dummy_input, export_path)
    
    # Log hyperparameters
    hparam_dict = {
        'game': args.game,
        'board_size': args.board_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'channels': args.channels,
        'res_blocks': args.res_blocks
    }
    
    metric_dict = {
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final models saved to {args.output_dir}")

if __name__ == "__main__":
    main()