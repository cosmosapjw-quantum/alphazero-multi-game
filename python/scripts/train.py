#!/usr/bin/env python3
"""
Training script for AlphaZero models.

This script trains an AlphaZero model using self-play game data.

Usage:
    python train.py [options]

Options:
    --data-dir DIR           Directory containing game data (default: data/games)
    --model-dir DIR          Directory to save models (default: models)
    --game {gomoku,chess,go} Game type (default: gomoku)
    --size SIZE              Board size (default: depends on game)
    --lr LR                  Learning rate (default: 0.001)
    --batch-size BATCH       Batch size (default: 256)
    --epochs EPOCHS          Number of epochs (default: 20)
    --checkpoint CHECK       Checkpoint frequency in epochs (default: 1)
    --resume MODEL           Resume training from model
    --variant               Use variant rules
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyalphazero as az
from alphazero.models import DDWRandWireResNet
from alphazero.training import AlphaZeroLoss, WarmupCosineAnnealingLR
from alphazero.training.dataset import GameDatasetBuilder, AlphaZeroDataset

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument("--data-dir", type=str, default="data/games",
                        help="Directory containing game data")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--checkpoint", type=int, default=1,
                        help="Checkpoint frequency in epochs")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from model")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--l2-reg", type=float, default=1e-4,
                        help="L2 regularization strength")
    parser.add_argument("--channels", type=int, default=128,
                        help="Number of channels in the model")
    parser.add_argument("--blocks", type=int, default=20,
                        help="Number of residual blocks")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for training")
    return parser.parse_args()


def prepare_dataset(args, game_type):
    """Prepare the dataset for training."""
    # Create dataset builder
    dataset_builder = GameDatasetBuilder(
        game_type, 
        use_enhanced_features=True, 
        include_augmentations=True
    )
    
    # Load games from directory
    print(f"Loading games from {args.data_dir}...")
    num_games = dataset_builder.add_games_from_directory(args.data_dir)
    print(f"Loaded {num_games} games")
    
    if num_games == 0:
        raise ValueError(f"No games found in {args.data_dir}")
    
    # Extract examples
    print("Extracting training examples...")
    examples = dataset_builder.extract_examples()
    print(f"Extracted {len(examples)} examples")
    
    # Create PyTorch dataset
    dataset = AlphaZeroDataset(examples)
    
    # Split into training and validation sets
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
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
    
    return train_loader, val_loader


def create_model(args, input_channels, action_size):
    """Create or load a model."""
    # Create model
    model = DDWRandWireResNet(
        input_channels=input_channels,
        output_size=action_size,
        channels=args.channels,
        num_blocks=args.blocks
    )
    
    # Load weights if resuming
    if args.resume:
        print(f"Loading model from {args.resume}")
        model.load_state_dict(torch.load(args.resume))
    
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_l2_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for states, policies, values in progress_bar:
        # Move to device
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_preds = model(states)
        
        # Calculate loss
        loss, policy_loss, value_loss, l2_loss = criterion(
            policy_logits, value_preds, policies, values, model
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update learning rate if using OneCycleLR
        if scheduler is not None and not isinstance(scheduler, (
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
            WarmupCosineAnnealingLR
        )):
            scheduler.step()
        
        # Update statistics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_l2_loss += l2_loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "p_loss": policy_loss.item(),
            "v_loss": value_loss.item()
        })
    
    # Calculate averages
    avg_loss = total_loss / batch_count
    avg_policy_loss = total_policy_loss / batch_count
    avg_value_loss = total_value_loss / batch_count
    avg_l2_loss = total_l2_loss / batch_count
    
    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "l2_loss": avg_l2_loss
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_l2_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for states, policies, values in dataloader:
            # Move to device
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Forward pass
            policy_logits, value_preds = model(states)
            
            # Calculate loss
            loss, policy_loss, value_loss, l2_loss = criterion(
                policy_logits, value_preds, policies, values, model
            )
            
            # Update statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_l2_loss += l2_loss.item()
            batch_count += 1
    
    # Calculate averages
    avg_loss = total_loss / batch_count
    avg_policy_loss = total_policy_loss / batch_count
    avg_value_loss = total_value_loss / batch_count
    avg_l2_loss = total_l2_loss / batch_count
    
    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "l2_loss": avg_l2_loss
    }


def run_training(args):
    """Run the training process."""
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
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
            board_size = 19  # Default to 19x19
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Create a test game state to get input shape
    game_state = az.createGameState(game_type, board_size, args.variant)
    tensor_rep = game_state.getEnhancedTensorRepresentation()
    input_channels = len(tensor_rep)
    action_size = game_state.getActionSpaceSize()
    
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Input channels: {input_channels}")
    print(f"Action space size: {action_size}")
    
    # Prepare dataset
    train_loader, val_loader = prepare_dataset(args, game_type)
    
    # Create model
    model = create_model(args, input_channels, action_size)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = AlphaZeroLoss(l2_reg=args.l2_reg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Create learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=args.warmup_epochs, 
            max_epochs=args.epochs,
            min_lr=args.lr * 0.01
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.epochs // 3, 
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Training history
    history = {
        "train_loss": [],
        "train_policy_loss": [],
        "train_value_loss": [],
        "val_loss": [],
        "val_policy_loss": [],
        "val_value_loss": []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None and isinstance(scheduler, (
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
            WarmupCosineAnnealingLR
        )):
            scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Policy Loss: {train_metrics['policy_loss']:.4f}, "
              f"Value Loss: {train_metrics['value_loss']:.4f}, "
              f"L2 Loss: {train_metrics['l2_loss']:.4f}")
        
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Policy Loss: {val_metrics['policy_loss']:.4f}, "
              f"Value Loss: {val_metrics['value_loss']:.4f}, "
              f"L2 Loss: {val_metrics['l2_loss']:.4f}")
        
        # Print epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {lr:.6f}")
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_policy_loss"].append(train_metrics["policy_loss"])
        history["train_value_loss"].append(train_metrics["value_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_policy_loss"].append(val_metrics["policy_loss"])
        history["val_value_loss"].append(val_metrics["value_loss"])
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint == 0:
            checkpoint_path = os.path.join(args.model_dir, f"{args.game}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_path = os.path.join(args.model_dir, f"{args.game}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
    
    # Print total training time
    total_time = time.time() - training_start_time
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f"{args.game}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save history
    history_path = os.path.join(args.model_dir, f"{args.game}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Save metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game": args.game,
        "board_size": board_size,
        "input_channels": input_channels,
        "action_size": action_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "l2_reg": args.l2_reg,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "model_channels": args.channels,
        "model_blocks": args.blocks,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "device": str(device),
        "variant_rules": args.variant
    }
    
    metadata_path = os.path.join(args.model_dir, f"{args.game}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history["train_policy_loss"], label="Train")
    plt.plot(history["val_policy_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Policy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history["train_value_loss"], label="Train")
    plt.plot(history["val_value_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Value Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.model_dir, f"{args.game}_training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    

if __name__ == "__main__":
    args = parse_args()
    run_training(args)