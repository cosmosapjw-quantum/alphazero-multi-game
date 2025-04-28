#!/usr/bin/env python3
"""
AlphaZero Training Script

This script trains a neural network for AlphaZero using data generated from self-play.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import glob
import random
from alphazero_model import DDWRandWireResNet, create_gomoku_model

class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training"""
    def __init__(self, data_dir: str, max_files: int = None):
        self.game_files = glob.glob(os.path.join(data_dir, "*.json"))
        if max_files is not None and max_files > 0:
            self.game_files = self.game_files[:max_files]
        
        self.examples = []
        self.load_data()
        
    def load_data(self):
        """Load data from game files"""
        for file_path in self.game_files:
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                # Extract training examples from the game
                examples = self.extract_examples(game_data)
                self.examples.extend(examples)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Shuffle examples
        random.shuffle(self.examples)
        print(f"Loaded {len(self.examples)} examples from {len(self.game_files)} games")
    
    def extract_examples(self, game_data: Dict) -> List[Dict]:
        """Extract training examples from a game"""
        examples = []
        game_type = game_data["game_type"]
        board_size = game_data["board_size"]
        result = game_data["result"]
        moves = game_data["moves"]
        
        # Recreate game states
        states = []
        current_state = self.create_initial_state(game_type, board_size)
        
        for move in moves:
            states.append(current_state.copy())
            current_state = self.update_state(current_state, move["action"])
        
        # Create examples with proper values based on final result
        for i, (state, move) in enumerate(zip(states, moves)):
            player = i % 2 + 1  # 1-based player index
            
            # Determine value target based on final result
            value = 0.0
            if result == 1 and player == 1:  # Player 1 wins, and this is player 1's move
                value = 1.0
            elif result == 1 and player == 2:  # Player 1 wins, but this is player 2's move
                value = -1.0
            elif result == 2 and player == 2:  # Player 2 wins, and this is player 2's move
                value = 1.0
            elif result == 2 and player == 1:  # Player 2 wins, but this is player 1's move
                value = -1.0
            
            examples.append({
                "state": state,
                "policy": move["policy"],
                "value": value
            })
        
        return examples
    
    def create_initial_state(self, game_type: int, board_size: int) -> np.ndarray:
        """Create initial state for a game"""
        if game_type == 0:  # Gomoku
            # For Gomoku, we'll use a simplistic representation for our training script
            # Our full implementation would use the enhanced tensor representation
            # Here, we'll use a minimal representation with 2 channels:
            # - Channel 0: Current player's stones
            # - Channel 1: Opponent's stones
            return np.zeros((2, board_size, board_size), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
    
    def update_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Update state with an action"""
        board_size = state.shape[1]
        x, y = action // board_size, action % board_size
        
        # Copy state
        new_state = state.copy()
        
        # Place stone for current player, then swap perspectives
        new_state[0, x, y] = 1.0
        
        # Swap perspectives
        new_state = np.array([new_state[1], new_state[0]])
        
        return new_state
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[idx]
        
        # Convert to tensors
        state_tensor = torch.from_numpy(example["state"]).float()
        policy_tensor = torch.tensor(example["policy"]).float()
        value_tensor = torch.tensor([example["value"]]).float()
        
        return state_tensor, policy_tensor, value_tensor

class AlphaZeroTrainer:
    """Trainer for AlphaZero"""
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device, 
                 value_weight: float = 1.0, policy_weight: float = 1.0, l2_weight: float = 1e-4):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.l2_weight = l2_weight
        
        self.train_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for states, policies, values in train_loader:
            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Calculate losses
            policy_loss = F.cross_entropy(policy_pred, policies)
            value_loss = F.mse_loss(value_pred, values)
            
            # Calculate L2 regularization
            l2_reg = 0.0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            # Total loss
            loss = (self.policy_weight * policy_loss + 
                   self.value_weight * value_loss + 
                   self.l2_weight * l2_reg)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * states.size(0)
            policy_loss_sum += policy_loss.item() * states.size(0)
            value_loss_sum += value_loss.item() * states.size(0)
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader.dataset)
        avg_policy_loss = policy_loss_sum / len(train_loader.dataset)
        avg_value_loss = value_loss_sum / len(train_loader.dataset)
        
        self.train_losses.append(avg_loss)
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        with torch.no_grad():
            for states, policies, values in val_loader:
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Calculate losses
                policy_loss = F.cross_entropy(policy_pred, policies)
                value_loss = F.mse_loss(value_pred, values)
                
                # Calculate L2 regularization
                l2_reg = 0.0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                
                # Total loss
                loss = (self.policy_weight * policy_loss + 
                       self.value_weight * value_loss + 
                       self.l2_weight * l2_reg)
                
                # Update statistics
                total_loss += loss.item() * states.size(0)
                policy_loss_sum += policy_loss.item() * states.size(0)
                value_loss_sum += value_loss.item() * states.size(0)
        
        # Calculate average losses
        avg_loss = total_loss / len(val_loader.dataset)
        avg_policy_loss = policy_loss_sum / len(val_loader.dataset)
        avg_value_loss = value_loss_sum / len(val_loader.dataset)
        
        self.val_losses.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }
    
    def save_checkpoint(self, path: str, epoch: int, lr: float) -> None:
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'val_losses': self.val_losses,
            'lr': lr
        }, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        return checkpoint

    def plot_loss(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 8))
        
        # Total losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Component losses
        plt.subplot(2, 1, 2)
        plt.plot(self.policy_losses, label='Policy Loss')
        plt.plot(self.value_losses, label='Value Loss')
        plt.legend()
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training Script")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing self-play game data")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save model checkpoints")
    parser.add_argument("--game", choices=["gomoku", "chess", "go"], default="gomoku",
                       help="Game type")
    parser.add_argument("--board-size", type=int, default=15,
                       help="Board size")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = AlphaZeroDataset(args.data_dir)
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.game == "gomoku":
        model = create_gomoku_model(args.board_size, args.seed)
    else:
        raise ValueError(f"Game type {args.game} not implemented yet")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = AlphaZeroTrainer(model, optimizer, device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = trainer.load_checkpoint(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f"  Train Loss: {train_metrics['loss']:.6f}, Policy Loss: {train_metrics['policy_loss']:.6f}, Value Loss: {train_metrics['value_loss']:.6f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"  Val Loss: {val_metrics['loss']:.6f}, Policy Loss: {val_metrics['policy_loss']:.6f}, Value Loss: {val_metrics['value_loss']:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"{args.game}_{epoch + 1:03d}.pt")
        trainer.save_checkpoint(checkpoint_path, epoch, args.lr)
        
        # Export model at regular intervals
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            export_path = os.path.join(args.output_dir, f"{args.game}_model_{epoch + 1:03d}.pt")
            model.eval()
            example_input = torch.randn(1, model.input_channels, model.board_size, model.board_size).to(device)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(export_path)
            print(f"Exported model to {export_path}")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, f"{args.game}_training_curves.png")
    trainer.plot_loss(plot_path)
    
    # Export final model
    final_path = os.path.join(args.output_dir, f"{args.game}_final.pt")
    model.eval()
    example_input = torch.randn(1, model.input_channels, model.board_size, model.board_size).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(final_path)
    print(f"Exported final model to {final_path}")

if __name__ == "__main__":
    main()