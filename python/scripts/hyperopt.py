#!/usr/bin/env python3
"""
Hyperparameter optimization script for AlphaZero models.

This script uses Optuna to find optimal hyperparameters for training AlphaZero models.

Usage:
    python hyperopt.py [options]

Options:
    --data-dir DIR           Directory containing game data (default: data/games)
    --output-dir DIR         Directory to save models and results (default: hyperopt_results)
    --game {gomoku,chess,go} Game type (default: gomoku)
    --size SIZE              Board size (default: depends on game)
    --trials NUM             Number of optimization trials (default: 20)
    --epochs EPOCHS          Number of epochs per trial (default: 10)
    --timeout HOURS          Timeout in hours (default: 24)
    --batch-size BATCH       Base batch size (default: 256)
    --variant                Use variant rules
    --gpu-ids GPU_IDS        Comma-separated list of GPU IDs to use (default: 0)
    --eval-games NUM         Number of evaluation games per trial (default: 10)
"""

import os
import sys
import argparse
import json
import time
import random
import subprocess
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from tqdm import tqdm
import pyalphazero as az
from alphazero.utils.elo import EloRating

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Hyperparameter Optimization")
    parser.add_argument("--data-dir", type=str, default="data/games",
                        help="Directory containing game data")
    parser.add_argument("--output-dir", type=str, default="hyperopt_results",
                        help="Directory to save models and results")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of optimization trials")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per trial")
    parser.add_argument("--timeout", type=float, default=24.0,
                        help="Timeout in hours")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Base batch size")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--eval-games", type=int, default=10,
                        help="Number of evaluation games per trial")
    parser.add_argument("--base-model", type=str, default="",
                        help="Base model for comparison (if empty, compare against random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval-simulations", type=int, default=800,
                        help="Number of MCTS simulations during evaluation")
    parser.add_argument("--resume-study", type=str, default="",
                        help="Resume an existing study (provide study name)")
    return parser.parse_args()


def create_model_for_training(input_channels, action_size, trial=None):
    """Create a model with the given hyperparameters for training."""
    # Import model class
    from alphazero.models import DDWRandWireResNet
    
    # Set model hyperparameters
    if trial:
        channels = trial.suggest_categorical("channels", [64, 96, 128, 160, 192, 256])
        num_blocks = trial.suggest_int("num_blocks", 10, 30, step=5)
    else:
        # Default values if no trial is provided
        channels = 128
        num_blocks = 20
    
    # Create model
    model = DDWRandWireResNet(
        input_channels=input_channels,
        output_size=action_size,
        channels=channels,
        num_blocks=num_blocks
    )
    
    return model


def train_model(trial, args, gpu_id=0):
    """Train a model with the given hyperparameters."""
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed + (trial.number if trial else 0))
        np.random.seed(args.seed + (trial.number if trial else 0))
        torch.manual_seed(args.seed + (trial.number if trial else 0))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed + (trial.number if trial else 0))
    
    # Set game-specific parameters
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Set board size
    if args.size <= 0:
        if args.game == "gomoku":
            board_size = 15
        elif args.game == "chess":
            board_size = 8
        elif args.game == "go":
            board_size = 9  # Smaller for faster training
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Create a test game state to get input shape
    game_state = az.createGameState(game_type, board_size, args.variant)
    tensor_rep = game_state.getEnhancedTensorRepresentation()
    input_channels = len(tensor_rep)
    action_size = game_state.getActionSpaceSize()
    
    # Create model directory
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number if trial else 'final'}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Suggest hyperparameters
    if trial:
        # Learning rate
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        # L2 regularization
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
        
        # Scheduler
        scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "step", "none"])
        
        # Warmup epochs (only if using cosine scheduler)
        if scheduler_type == "cosine":
            warmup_epochs = trial.suggest_int("warmup_epochs", 1, min(5, args.epochs))
        else:
            warmup_epochs = 0
        
        # Batch size multiplier
        batch_size_mult = trial.suggest_categorical("batch_size_mult", [0.5, 1.0, 2.0])
        batch_size = int(args.batch_size * batch_size_mult)
    else:
        # Default values if no trial is provided
        lr = 0.001
        l2_reg = 1e-4
        scheduler_type = "cosine"
        warmup_epochs = 3
        batch_size = args.batch_size
    
    # Log hyperparameters
    params = {
        "game": args.game,
        "board_size": board_size,
        "input_channels": input_channels,
        "action_size": action_size,
        "lr": lr,
        "l2_reg": l2_reg,
        "scheduler": scheduler_type,
        "warmup_epochs": warmup_epochs,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "device": str(device)
    }
    
    if trial:
        params.update({
            "channels": trial.params["channels"],
            "num_blocks": trial.params["num_blocks"]
        })
    
    # Save hyperparameters
    with open(os.path.join(trial_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    
    # Create model
    model = create_model_for_training(input_channels, action_size, trial)
    model.to(device)
    
    # Set training command arguments
    train_args = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--data-dir", args.data_dir,
        "--model-dir", trial_dir,
        "--game", args.game,
        "--size", str(board_size),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--epochs", str(args.epochs),
        "--l2-reg", str(l2_reg),
        "--scheduler", scheduler_type,
        "--use-gpu"
    ]
    
    if warmup_epochs > 0:
        train_args.extend(["--warmup-epochs", str(warmup_epochs)])
    
    if args.variant:
        train_args.append("--variant")
    
    if args.seed is not None:
        train_args.extend(["--seed", str(args.seed + (trial.number if trial else 0))])
    
    # Set environment variables for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Run training script
    print("\nRunning training process with the following command:")
    print(" ".join(train_args))
    
    start_time = time.time()
    
    training_process = subprocess.Popen(
        train_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor the output
    while True:
        line = training_process.stdout.readline()
        if not line and training_process.poll() is not None:
            break
        if line:
            print(line.rstrip())
    
    # Check for errors
    return_code = training_process.wait()
    if return_code != 0:
        error_output = training_process.stderr.read()
        print(f"Training failed with code {return_code}:")
        print(error_output)
        if trial:
            trial.report(-float('inf'), args.epochs)  # Report failure to Optuna
            raise optuna.exceptions.TrialPruned("Training failed")
        return None
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Return path to best model
    best_model_path = os.path.join(trial_dir, f"{args.game}_best.pt")
    
    # Check if model was successfully created
    if not os.path.exists(best_model_path):
        print(f"Warning: Best model not found at {best_model_path}")
        best_model_path = os.path.join(trial_dir, f"{args.game}_final.pt")
        if not os.path.exists(best_model_path):
            print(f"Error: Final model not found at {best_model_path}")
            if trial:
                trial.report(-float('inf'), args.epochs)  # Report failure to Optuna
                raise optuna.exceptions.TrialPruned("Model not found")
            return None
    
    return best_model_path


def evaluate_model(model_path, base_model_path, args, gpu_id=0):
    """Evaluate the trained model against a baseline model."""
    # Set environment variables for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set evaluation command arguments
    eval_args = [
        sys.executable, os.path.join(os.path.dirname(__file__), "evaluate.py"),
        "--model-a", model_path,
        "--model-b", base_model_path if base_model_path else "",  # Empty for random policy
        "--game", args.game,
        "--size", str(args.size),
        "--games", str(args.eval_games),
        "--simulations", str(args.eval_simulations),
        "--threads", "4",  # Fixed for consistency
        "--output-dir", args.output_dir,
        "--temperature", "0.1",  # Lower temperature for stronger play
        "--use-gpu"
    ]
    
    if args.variant:
        eval_args.append("--variant")
    
    # Run evaluation script
    print("\nRunning evaluation process with the following command:")
    print(" ".join(eval_args))
    
    start_time = time.time()
    
    eval_process = subprocess.Popen(
        eval_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Process the output to get the win rate
    output_lines = []
    win_rate = 0.0
    elo_diff = 0.0
    
    for line in eval_process.stdout:
        output_lines.append(line.strip())
        print(line.strip())
        
        # Extract win rate
        if "Evaluation Results" in line:
            # Next line should contain the model A info with win rate
            model_a_line = eval_process.stdout.readline().strip()
            output_lines.append(model_a_line)
            print(model_a_line)
            
            # Extract win rate
            try:
                win_rate = float(model_a_line.split("(")[1].split("%")[0])
            except:
                pass
            
            # Extract ELO rating
            try:
                elo_rating = float(model_a_line.split("ELO: ")[1].split(")")[0])
                # Get model B line for its ELO
                model_b_line = eval_process.stdout.readline().strip()
                output_lines.append(model_b_line)
                print(model_b_line)
                elo_b = float(model_b_line.split("ELO: ")[1].split(")")[0])
                elo_diff = elo_rating - elo_b
            except:
                pass
    
    # Check for errors
    return_code = eval_process.wait()
    if return_code != 0:
        error_output = eval_process.stderr.read()
        print(f"Evaluation failed with code {return_code}:")
        print(error_output)
        return 0.0  # Return lowest possible score
    
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Return the win rate as the objective value
    # If we have ELO difference, use that instead
    if elo_diff != 0.0:
        return elo_diff
    return win_rate


def objective(trial, args, gpu_id=0):
    """Optuna objective function for hyperparameter optimization."""
    print(f"\n\n=== Trial {trial.number} ===")
    
    # Train model with suggested hyperparameters
    model_path = train_model(trial, args, gpu_id)
    
    if model_path is None:
        return -float('inf')  # Training failed
    
    # Evaluate the model
    objective_value = evaluate_model(model_path, args.base_model, args, gpu_id)
    
    # Store the results
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("objective_value", objective_value)
    
    return objective_value


def run_hyperopt(args):
    """Run hyperparameter optimization."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up GPU IDs
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(",")]
    
    # Create or load the study
    study_name = f"alphazero_{args.game}_hyperopt"
    if args.resume_study:
        study_name = args.resume_study
    
    storage_name = f"sqlite:///{os.path.join(args.output_dir, 'optuna.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    # Function to assign GPU ID based on trial number
    def gpu_id_for_trial(trial):
        return gpu_ids[trial.number % len(gpu_ids)]
    
    # Run optimization
    timeout_seconds = args.timeout * 3600 if args.timeout > 0 else None
    
    study.optimize(
        lambda trial: objective(trial, args, gpu_id_for_trial(trial)),
        n_trials=args.trials,
        timeout=timeout_seconds
    )
    
    # Print best parameters
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print(f"  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Create visualizations
    try:
        # Plot optimization history
        history_fig = plot_optimization_history(study)
        history_fig.write_image(os.path.join(args.output_dir, "optimization_history.png"))
        
        # Plot parameter importances
        param_fig = plot_param_importances(study)
        param_fig.write_image(os.path.join(args.output_dir, "param_importances.png"))
    except:
        print("Warning: Failed to generate visualization plots")
    
    # Save study results
    study_results = {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "best_model_path": best_trial.user_attrs.get("model_path", ""),
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "model_path": t.user_attrs.get("model_path", ""),
                "state": t.state.name
            }
            for t in study.trials
        ]
    }
    
    with open(os.path.join(args.output_dir, "hyperopt_results.json"), "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Train a final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    
    # Set trial object with best parameters
    class BestTrial:
        def __init__(self, params):
            self.params = params
            self.number = 999  # Special number for final model
        
        def suggest_categorical(self, name, choices):
            return self.params[name]
        
        def suggest_int(self, name, low, high, step=1):
            return self.params[name]
        
        def suggest_float(self, name, low, high, log=False):
            return self.params[name]
    
    best_trial_obj = BestTrial(best_trial.params)
    final_model_path = train_model(best_trial_obj, args, gpu_ids[0])
    
    if final_model_path:
        print(f"\nFinal model trained and saved to {final_model_path}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_score = evaluate_model(final_model_path, args.base_model, args, gpu_ids[0])
        print(f"\nFinal model performance: {final_score}")
        
        # Update results file
        study_results["final_model_path"] = final_model_path
        study_results["final_model_score"] = final_score
        
        with open(os.path.join(args.output_dir, "hyperopt_results.json"), "w") as f:
            json.dump(study_results, f, indent=2)
    else:
        print("\nFailed to train final model")
    
    return study


if __name__ == "__main__":
    args = parse_args()
    study = run_hyperopt(args)
    
    print("\nHyperparameter optimization completed successfully!")
    print(f"Results saved to {args.output_dir}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")