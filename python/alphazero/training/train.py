# python/alphazero/training/train.py
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import _alphazero_cpp as az
from alphazero.models.ddw_randwire import DDWRandWireResNet
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alphazero.training")

class AlphaZeroLoss(nn.Module):
    def __init__(self, l2_reg=1e-4):
        super(AlphaZeroLoss, self).__init__()
        self.l2_reg = l2_reg
    
    def forward(self, policy_output, value_output, policy_target, value_target, model):
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_output, policy_target)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_output.squeeze(-1), value_target)
        
        # L2 regularization
        l2_reg_loss = torch.tensor(0.0, requires_grad=True, device=policy_output.device)
        for param in model.parameters():
            l2_reg_loss = l2_reg_loss + torch.norm(param)**2
        
        # Combined loss
        total_loss = policy_loss + value_loss + self.l2_reg * l2_reg_loss
        
        return total_loss, policy_loss, value_loss, l2_reg_loss

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Training Script")
    parser.add_argument("--game", type=str, required=True, choices=["gomoku", "chess", "go"],
                        help="Game to train on")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing training data")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--board-size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--l2-reg", type=float, default=1e-4,
                        help="L2 regularization weight")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["step", "cosine", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--channels", type=int, default=128,
                        help="Number of channels in the network")
    parser.add_argument("--blocks", type=int, default=20,
                        help="Number of residual blocks")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for training")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to model to resume training from")
    parser.add_argument("--self-play", action="store_true",
                        help="Generate self-play data before training")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of self-play games to generate")
    parser.add_argument("--mcts-sims", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--mcts-threads", type=int, default=4,
                        help="Number of MCTS threads")
    parser.add_argument("--variant-rules", action="store_true",
                        help="Use variant rules (Renju for Gomoku, Chess960, etc.)")
    return parser.parse_args()

def prepare_dataset(args):
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Create dataset
    dataset = az.Dataset()
    
    # If self-play is enabled, generate new games
    if args.self_play:
        logger.info(f"Generating {args.num_games} self-play games...")
        
        # Create a model for self-play
        device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Get input shape and action size
        game_state = az.createGameState(game_type, args.board_size, args.variant_rules)
        
        if args.board_size == 0:
            board_size = game_state.getBoardSize()
        else:
            board_size = args.board_size
        
        tensor_shape = game_state.getEnhancedTensorRepresentation()
        input_channels = len(tensor_shape)
        action_size = game_state.getActionSpaceSize()
        
        # Create or load model
        if args.resume:
            model = DDWRandWireResNet(input_channels, action_size, args.channels, args.blocks)
            model.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            model = DDWRandWireResNet(input_channels, action_size, args.channels, args.blocks)
        
        model = model.to(device)
        model.eval()
        
        # Create neural network wrapper
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
                    policy = F.softmax(policy_logits, dim=1)[0].cpu().numpy()
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
                    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                    value_list = value_tensor.squeeze(-1).cpu().numpy()
                
                # Set output
                for i in range(batch_size):
                    policies[i] = policy_probs[i].tolist()
                    values[i] = value_list[i]
            
            def predictAsync(self, state):
                # Just call predict for now
                return self.predict(state)
            
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
                return 64
            
            def getModelInfo(self):
                return "DDWRandWireResNet"
            
            def getModelSizeBytes(self):
                return sum(p.numel() * 4 for p in self.model.parameters())
            
            def benchmark(self, numIterations=100, batchSize=16):
                pass
            
            def enableDebugMode(self, enable):
                pass
            
            def printModelSummary(self):
                print(self.model)
        
        # Create neural network wrapper
        nn_wrapper = TorchNeuralNetwork(model, device)
        
        # Create self-play manager
        manager = az.SelfPlayManager(nn_wrapper, args.num_games, args.mcts_sims, args.mcts_threads)
        
        # Set exploration parameters
        manager.setExplorationParams(
            dirichletAlpha=0.03 if args.game != "chess" else 0.3,
            dirichletEpsilon=0.25,
            initialTemperature=1.0,
            temperatureDropMove=30,
            finalTemperature=0.0
        )
        
        # Set up game saving
        os.makedirs(os.path.join(args.data_dir, "games"), exist_ok=True)
        manager.setSaveGames(True, os.path.join(args.data_dir, "games"))
        
        # Generate games
        start_time = time.time()
        games = manager.generateGames(game_type, args.board_size, args.variant_rules)
        end_time = time.time()
        
        logger.info(f"Generated {len(games)} games in {end_time - start_time:.2f} seconds")
        
        # Add games to dataset
        for game in games:
            dataset.addGameRecord(game)
    else:
        # Load data from files
        game_dir = os.path.join(args.data_dir, "games")
        if os.path.exists(game_dir):
            logger.info(f"Loading games from {game_dir}...")
            for filename in os.listdir(game_dir):
                if filename.endswith(".json"):
                    try:
                        game = az.GameRecord.loadFromFile(os.path.join(game_dir, filename))
                        dataset.addGameRecord(game)
                    except Exception as e:
                        logger.error(f"Error loading game {filename}: {e}")
    
    # Extract training examples
    logger.info("Extracting training examples...")
    dataset.extractExamples(True)
    logger.info(f"Dataset size: {dataset.size()} examples")
    
    return dataset

def create_model(args, input_channels, action_size):
    # Create model
    model = DDWRandWireResNet(
        input_channels=input_channels,
        output_size=action_size,
        channels=args.channels,
        num_blocks=args.blocks
    )
    
    # Load weights if resuming
    if args.resume:
        logger.info(f"Loading model from {args.resume}")
        model.load_state_dict(torch.load(args.resume))
    
    return model

def train_epoch(model, dataloader, optimizer, criterion, device):
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

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare dataset
    dataset = prepare_dataset(args)
    
    if dataset.size() == 0:
        logger.error("No training examples found. Please generate or provide game data.")
        return
    
    # Get a sample to determine input shape
    sample = dataset.getRandomSubset(1)[0]
    input_channels = len(sample.state)
    
    # Convert game type to enum
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Create temporary game state to get action space size
    game_state = az.createGameState(game_type, args.board_size, args.variant_rules)
    action_size = game_state.getActionSpaceSize()
    
    logger.info(f"Input channels: {input_channels}")
    logger.info(f"Action space size: {action_size}")
    
    # Create model
    model = create_model(args, input_channels, action_size)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = AlphaZeroLoss(l2_reg=args.l2_reg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Create learning rate scheduler
    if args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # Create PyTorch Dataset
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.examples = dataset.getRandomSubset(dataset.size())
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            
            # Convert to PyTorch tensors
            state = torch.FloatTensor(example.state)
            policy = torch.FloatTensor(example.policy)
            value = torch.FloatTensor([example.value])
            
            return state, policy, value
    
    tensor_dataset = TensorDataset(dataset)
    
    # Split into train and validation sets
    dataset_size = len(tensor_dataset)
    val_size = min(1000, dataset_size // 10)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        tensor_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Training loop
    logger.info("Starting training")
    
    history = {
        "train_loss": [],
        "train_policy_loss": [],
        "train_value_loss": [],
        "val_loss": [],
        "val_policy_loss": [],
        "val_value_loss": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Policy Loss: {train_metrics['policy_loss']:.4f}, "
                    f"Value Loss: {train_metrics['value_loss']:.4f}")
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Policy Loss: {val_metrics['policy_loss']:.4f}, "
                    f"Value Loss: {val_metrics['value_loss']:.4f}")
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_policy_loss"].append(train_metrics["policy_loss"])
        history["train_value_loss"].append(train_metrics["value_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_policy_loss"].append(val_metrics["policy_loss"])
        history["val_value_loss"].append(val_metrics["value_loss"])
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_path = os.path.join(args.model_dir, f"{args.game}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.model_dir, f"{args.game}_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save history
        history_path = os.path.join(args.model_dir, f"{args.game}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f"{args.game}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history["train_policy_loss"], label="Train")
    plt.plot(history["val_policy_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Policy Loss")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history["train_value_loss"], label="Train")
    plt.plot(history["val_value_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Value Loss")
    plt.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.model_dir, f"{args.game}_training_history.png")
    plt.savefig(plot_path)
    logger.info(f"Saved training history plot to {plot_path}")

if __name__ == "__main__":
    main()