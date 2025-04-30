#!/usr/bin/env python3
"""
Self-play data generation for AlphaZero training.

This script generates self-play games using the specified model, ensuring that
neural network inference occurs within the C++ boundary via LibTorch models
to avoid GIL limitations and maximize performance. It supports automatic
conversion of PyTorch models to LibTorch format.

Usage:
    python self_play.py [options]

Options:
    --model MODEL         Path to model file (LibTorch .pt or PyTorch .pth/.pt)
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE           Board size (default: depends on game)
    --num-games NUM       Number of games to generate (default: 100)
    --simulations SIMS    Number of MCTS simulations per move (default: 800)
    --threads THREADS     Number of threads (default: 4)
    --output-dir DIR      Output directory (default: data/games)
    --temperature TEMP    Initial temperature (default: 1.0)
    --temp-drop MOVE      Move to drop temperature (default: 30)
    --final-temp TEMP     Final temperature (default: 0.0)
    --dirichlet-alpha A   Dirichlet noise alpha (default: 0.03)
    --dirichlet-epsilon E Dirichlet noise weight (default: 0.25)
    --variant             Use variant rules (Renju, Chess960, Chinese)
    --batch-size SIZE     Batch size for neural network inference (default: 8)
    --batch-timeout MS    Timeout for batch completion in milliseconds (default: 100)
    --no-gpu              Disable GPU acceleration
    --no-batched-search   Disable batched MCTS search
    --fp16                Use FP16 precision (faster but less accurate)
    --create-random-model Create and export a random model if no model is provided
"""

import os
import sys
import argparse
import time
import json
import random
import torch
import numpy as np
import tempfile
import threading
import atexit

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the build directory for the C++ extension
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'pybind'))
sys.path.insert(0, build_dir)

try:
    from alphazero.models import DDWRandWireResNet
    import _alphazero_cpp as az
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the project is properly built and the Python package is installed.")
    sys.exit(1)

# Keep track of temporary files to clean up
_temp_files = []

def _cleanup_temp_files():
    for temp_file in _temp_files:
        try:
            os.remove(temp_file)
            print(f"Cleaned up temporary model file: {temp_file}")
        except OSError:
            pass # Ignore errors during cleanup

atexit.register(_cleanup_temp_files)

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play with C++ Inference")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file (LibTorch or PyTorch)")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=12,
                        help="Number of threads")
    parser.add_argument("--output-dir", type=str, default="data/games",
                        help="Output directory for game records")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial temperature")
    parser.add_argument("--temp-drop", type=int, default=30,
                        help="Move number to drop temperature")
    parser.add_argument("--final-temp", type=float, default=0.0,
                        help="Final temperature")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03,
                        help="Dirichlet noise alpha parameter")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # GPU and batching related arguments
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for neural network inference")
    parser.add_argument("--batch-timeout", type=int, default=100,
                        help="Timeout for batch completion in milliseconds")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--no-batched-search", action="store_true",
                        help="Disable batched MCTS search")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision (faster but less accurate)")
    parser.add_argument("--create-random-model", action="store_true",
                        help="Create and export a random model if no model is provided")

    return parser.parse_args()


def export_pytorch_to_libtorch(model, input_shape, output_path):
    """Export a PyTorch model to LibTorch format for use with C++ API.

    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (batch_size, channels, height, width)
        output_path: Path to save the exported model

    Returns:
        Path to the exported model
    """
    print(f"Exporting PyTorch model to LibTorch format: {output_path}")

    # Ensure model is in eval mode
    model.eval()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a dummy input tensor
    dummy_input = torch.zeros(input_shape, dtype=torch.float32).to(device)

    # Trace the model
    try:
        print("Tracing model...")
        traced_model = torch.jit.trace(model, dummy_input)

        # Test the traced model
        print("Testing traced model...")
        with torch.no_grad():
            traced_output = traced_model(dummy_input)
            model_output = model(dummy_input)

            # Check if outputs match
            policy_match = torch.allclose(traced_output[0], model_output[0], atol=1e-5)
            value_match = torch.allclose(traced_output[1], model_output[1], atol=1e-5)

            if not policy_match or not value_match:
                print("WARNING: Traced model outputs don't match original model!")
                print(f"  Policy match: {policy_match}")
                print(f"  Value match: {value_match}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the traced model
        traced_model.save(output_path)
        print(f"Model successfully exported to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to export model: {e}")
        raise


def create_random_model(input_channels, action_size, board_size):
    """Create a random model for testing and export.

    Args:
        input_channels: Number of input channels
        action_size: Size of action space
        board_size: Board size

    Returns:
        PyTorch model
    """
    print("Creating a random DDWRandWireResNet model...")
    model = DDWRandWireResNet(input_channels, action_size, channels=64, num_blocks=8)

    # Initialize randomly
    for param in model.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

    model.eval()
    return model


def create_neural_network(args, game_type, board_size):
    """Create a neural network using the C++ API, potentially exporting a PyTorch model first.

    Args:
        args: Command-line arguments
        game_type: Game type enum value
        board_size: Board size

    Returns:
        C++ Neural network object (az.NeuralNetwork) or None if failed/not specified.
    """
    # Determine if GPU should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu

    # Create a test game state to get input shape and action size
    game_state = az.createGameState(game_type, board_size, args.variant)
    tensor_rep = game_state.getEnhancedTensorRepresentation()
    input_channels = len(tensor_rep)
    action_size = game_state.getActionSpaceSize()
    input_shape = (1, input_channels, board_size, board_size)

    model_path_to_load = args.model
    is_temp_model = False

    # Handle creation of a random model if requested and no model provided
    if args.create_random_model and not args.model:
        print("Creating and exporting a random model...")
        try:
            # Create a random PyTorch model
            pytorch_model = create_random_model(input_channels, action_size, board_size)

            # Define export path
            export_dir = "models"
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, f"random_model_{args.game}_{board_size}x{board_size}.pt")

            # Export to LibTorch
            model_path_to_load = export_pytorch_to_libtorch(pytorch_model, input_shape, export_path)
            print(f"Random model exported to {model_path_to_load}")
            print("You can now use this model for subsequent runs with:")
            print(f"  python self_play.py --model {model_path_to_load} ...")

        except Exception as e:
            print(f"Failed to create or export random model: {e}")
            print("Proceeding without a neural network (random policy).")
            return None

    # If a model path is provided, attempt to load it
    if model_path_to_load:
        nn = None
        # 1. Try loading directly with C++ API (assumes LibTorch format)
        try:
            print(f"Attempting to load model directly with C++ API: {model_path_to_load}")
            # Set batch size and timeout for C++ network
            az.NeuralNetwork.set_batch_size(args.batch_size)
            az.NeuralNetwork.set_batch_timeout(args.batch_timeout)
            az.NeuralNetwork.set_use_fp16(args.fp16)

            nn = az.createNeuralNetwork(model_path_to_load, game_type, board_size, use_gpu)
            print(f"Successfully loaded model using C++ API from: {model_path_to_load}")

        except Exception as e_cpp:
            print(f"Failed to load model with C++ API: {e_cpp}")
            print("Assuming it might be a PyTorch model, attempting conversion...")

            # 2. If C++ loading fails, try loading as PyTorch and exporting
            try:
                device = torch.device("cuda" if use_gpu else "cpu")
                pytorch_model = DDWRandWireResNet(input_channels, action_size) # Adapt if model architecture varies
                pytorch_model.load_state_dict(torch.load(model_path_to_load, map_location=device))
                pytorch_model.to(device)
                pytorch_model.eval()
                print(f"Successfully loaded model using PyTorch: {model_path_to_load}")

                # Export to a temporary LibTorch file
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                    temp_libtorch_path = tmp_file.name

                print(f"Exporting PyTorch model to temporary LibTorch file: {temp_libtorch_path}")
                export_pytorch_to_libtorch(pytorch_model, input_shape, temp_libtorch_path)

                # Keep track of the temp file for cleanup
                _temp_files.append(temp_libtorch_path)
                is_temp_model = True
                model_path_to_load = temp_libtorch_path

                # Try loading the exported temporary model with C++ API again
                print(f"Attempting to load exported temporary model with C++ API: {model_path_to_load}")
                # Set batch size and timeout again if needed (might be reset)
                az.NeuralNetwork.set_batch_size(args.batch_size)
                az.NeuralNetwork.set_batch_timeout(args.batch_timeout)
                az.NeuralNetwork.set_use_fp16(args.fp16)
                nn = az.createNeuralNetwork(model_path_to_load, game_type, board_size, use_gpu)
                print(f"Successfully loaded exported model using C++ API from: {model_path_to_load}")

            except Exception as e_export:
                print(f"Failed to load with PyTorch or export to LibTorch: {e_export}")
                if is_temp_model and os.path.exists(model_path_to_load):
                    os.remove(model_path_to_load) # Clean up failed temp export
                print("Failed to load or convert the model.")
                print("Proceeding without a neural network (random policy).")
                return None

        # If loading succeeded (either directly or via export)
        if nn:
            if use_gpu:
                print(f"Using GPU acceleration with device: {nn.getDeviceInfo()}")
            else:
                print(f"Using CPU: {nn.getDeviceInfo()}")

            print(f"C++ NN Batch size: {nn.getBatchSize()}")
            print(f"C++ NN Batch timeout: {nn.getBatchTimeout()} ms")
            print(f"C++ NN FP16 enabled: {nn.getUseFp16()}")
            if nn.getInferenceTimeMs() > 0: # May be 0 initially
                 print(f"Average inference time: {nn.getInferenceTimeMs():.2f} ms (will update)")
            print("Neural network loaded via C++ API - GIL issues avoided.")
            return nn
        else:
             # Should not happen if logic is correct, but as a safeguard
            print("Model loading failed unexpectedly.")
            print("Proceeding without a neural network (random policy).")
            return None

    # If no model path was provided and not creating a random one
    print("No model specified. Using random policy network (C++ internal).")
    return None


def run_self_play(args):
    """Run self-play games using the C++ SelfPlayManager.

    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        # Also seed the C++ random generator if possible (assuming a function exists)
        try:
            az.set_random_seed(args.seed)
            print("C++ random seed set.")
        except AttributeError:
            print("Warning: Could not set C++ random seed (az.set_random_seed not found).")


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
            # Should not happen due to choices in argparse
            print(f"Warning: Unknown game type '{args.game}', defaulting size to 15.")
            board_size = 15
    else:
        board_size = args.size

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine if GPU and batched search should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    # Note: C++ SelfPlayManager likely handles batched search internally if NN supports batching
    use_batched_search = use_gpu and not args.no_batched_search # Keep for info, C++ decides

    # Create neural network (always attempts C++ path)
    print("Initializing Neural Network...")
    neural_network = create_neural_network(args, game_type, board_size)

    # Create self-play manager using the C++ implementation
    print("Initializing Self-Play Manager...")
    try:
        self_play = az.SelfPlayManager(
            neural_network, # Can be None for random policy
            args.num_games,
            args.simulations,
            args.threads
        )
    except Exception as e:
        print(f"Fatal error creating SelfPlayManager: {e}")
        sys.exit(1)

    # Set exploration parameters
    self_play.setExplorationParams(
        args.dirichlet_alpha,
        args.dirichlet_epsilon,
        args.temperature,
        args.temp_drop,
        args.final_temp
    )

    # Set game saving parameters
    self_play.setSaveGames(True, args.output_dir)

    # Set game type and variant for the manager
    self_play.setGameParams(game_type, board_size, args.variant)

    # --- Start Game Generation ---
    print(f"Starting self-play generation...")
    print("-" * 30)
    print(f"Game:               {args.game.upper()}")
    print(f"Board size:         {board_size}x{board_size}")
    print(f"Variant rules:      {args.variant}")
    print(f"Number of games:    {args.num_games}")
    print(f"Simulations/move:   {args.simulations}")
    print(f"Number of threads:  {args.threads}")
    print(f"Using GPU:          {use_gpu}")
    print(f"Using FP16:         {args.fp16 and use_gpu}")
    print(f"NN Batch Size:      {args.batch_size if neural_network else 'N/A'}")
    print(f"NN Batch Timeout:   {args.batch_timeout if neural_network else 'N/A'} ms")
    print(f"Output directory:   {args.output_dir}")
    print(f"Model path:         {args.model if args.model else 'Random (C++)'}")
    if neural_network:
        print(f"NN Device Info:     {neural_network.getDeviceInfo()}")
    print("-" * 30)

    # Performance monitoring variables
    start_time = time.time()
    last_update_time = start_time
    completed_games_count = 0
    total_moves_count = 0

    try:
        # Use the C++ manager's progress callback mechanism if available
        # Assuming a hypothetical structure like this:
        if hasattr(self_play, 'setProgressCallback'):
            def progress_update(completed_games, total_moves):
                nonlocal last_update_time, completed_games_count, total_moves_count
                current_time = time.time()
                elapsed_since_last = current_time - last_update_time
                games_since_last = completed_games - completed_games_count

                if elapsed_since_last > 0 and games_since_last > 0:
                    rate = games_since_last / elapsed_since_last
                    print(f"Progress: {completed_games}/{args.num_games} games | "
                          f"{total_moves} moves | Rate: {rate:.2f} games/sec")
                else:
                     print(f"Progress: {completed_games}/{args.num_games} games | "
                           f"{total_moves} moves")

                last_update_time = current_time
                completed_games_count = completed_games
                total_moves_count = total_moves

            # Register callback to be called every N seconds or M games from C++
            self_play.setProgressCallback(progress_update, interval_seconds=10)
            print("Registered progress callback.")

            # Blocking call to generate games
            self_play.generateGames() # Assumes generateGames is now blocking and uses callback

        else:
            # Fallback if no callback: manual blocking call and stats after
            print("Progress callback not available in C++ module. Stats will be shown at the end.")
            # This assumes generateGames returns completed games list, adjust if it doesn't
            games_data = self_play.generateGames() # Might return game objects or just stats
            # If it returns game objects (like az.GameRecord), process them:
            if isinstance(games_data, list) and games_data and hasattr(games_data[0], 'getMoves'):
                 completed_games_count = len(games_data)
                 total_moves_count = sum(len(game.getMoves()) for game in games_data)
            elif isinstance(games_data, dict): # Or if it returns a stats dictionary
                 completed_games_count = games_data.get('completed_games', 0)
                 total_moves_count = games_data.get('total_moves', 0)
            else: # Or just rely on the manager's internal count if available
                 completed_games_count = self_play.getCompletedGamesCount() if hasattr(self_play, 'getCompletedGamesCount') else 0
                 total_moves_count = self_play.getTotalMovesCount() if hasattr(self_play, 'getTotalMovesCount') else 0


    except KeyboardInterrupt:
        print("Self-play interrupted by user.")
        # Attempt to get final counts from the manager if possible
        if hasattr(self_play, 'getCompletedGamesCount'):
             completed_games_count = self_play.getCompletedGamesCount()
        if hasattr(self_play, 'getTotalMovesCount'):
             total_moves_count = self_play.getTotalMovesCount()
        print("Processing results from completed games (if any)...")
    except Exception as e:
        print(f"An error occurred during self-play: {e}")
        # Attempt to get final counts
        if hasattr(self_play, 'getCompletedGamesCount'):
             completed_games_count = self_play.getCompletedGamesCount()
        if hasattr(self_play, 'getTotalMovesCount'):
             total_moves_count = self_play.getTotalMovesCount()
        print("Processing results from completed games (if any)...")


    end_time = time.time()
    total_duration = end_time - start_time

    # --- Print Results ---
    print("--- Self-Play Results ---")
    print(f"Completed {completed_games_count} games in {total_duration:.2f} seconds")

    if completed_games_count == 0:
        print("No complete games were generated.")
        return # Exit early if no games

    # Calculate final statistics
    avg_moves_per_game = total_moves_count / completed_games_count if completed_games_count > 0 else 0
    avg_time_per_game = total_duration / completed_games_count if completed_games_count > 0 else 0
    avg_moves_per_second = total_moves_count / total_duration if total_duration > 0 else 0

    print(f"Total moves:        {total_moves_count}")
    print(f"Avg moves/game:     {avg_moves_per_game:.1f}")
    print(f"Avg time/game:      {avg_time_per_game:.2f} seconds")
    print(f"Avg moves/second:   {avg_moves_per_second:.1f}")

    # Performance details if neural network was used
    if neural_network and hasattr(neural_network, 'getInferenceTimeMs'):
        avg_inference_time = neural_network.getInferenceTimeMs()
        if avg_inference_time > 0:
            print(f"Avg NN inference:   {avg_inference_time:.2f} ms (C++ API)")
            # Estimate NN contribution percentage (rough estimate)
            # total_nn_calls = total_moves * simulations
            # total_nn_time_ms = total_nn_calls * avg_inference_time
            # percentage_time = (total_nn_time_ms / 1000) / total_duration * 100
            # print(f"Est. NN time %:     {percentage_time:.1f}% (approx)")
        else:
            print("Avg NN inference:   N/A (no inferences recorded)")

    # --- Save Metadata ---
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game": args.game,
        "board_size": board_size,
        "num_games_requested": args.num_games,
        "num_games_completed": completed_games_count,
        "simulations": args.simulations,
        "threads": args.threads,
        "temperature": args.temperature,
        "temp_drop": args.temp_drop,
        "final_temp": args.final_temp,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "variant": args.variant,
        "model_path_arg": args.model, # Original arg
        "total_moves": total_moves_count,
        "avg_moves_per_game": avg_moves_per_game,
        "total_time_seconds": total_duration,
        "avg_time_per_game_seconds": avg_time_per_game,
        "avg_moves_per_second": avg_moves_per_second,
        "use_gpu": use_gpu,
        "fp16_used": args.fp16 and use_gpu, # Actual usage depends on NN load
        "batch_size_used": args.batch_size if neural_network else None,
        "batch_timeout_used": args.batch_timeout if neural_network else None,
        "seed": args.seed
    }

    # Add neural network specific info if available
    if neural_network:
        metadata["nn_loaded"] = True
        if hasattr(neural_network, 'getInferenceTimeMs'):
            metadata["nn_avg_inference_ms"] = neural_network.getInferenceTimeMs()
        if hasattr(neural_network, 'getDeviceInfo'):
            metadata["nn_device_info"] = neural_network.getDeviceInfo()
        # Assuming the C++ network object might have a way to report the actual model file loaded
        # or some internal identifier. Replace 'getLoadedModelIdentifier' with actual method if it exists.
        # metadata["nn_loaded_identifier"] = neural_network.getLoadedModelIdentifier()
        metadata["nn_batch_size"] = neural_network.getBatchSize() if hasattr(neural_network, 'getBatchSize') else args.batch_size
        metadata["nn_batch_timeout"] = neural_network.getBatchTimeout() if hasattr(neural_network, 'getBatchTimeout') else args.batch_timeout
        metadata["nn_fp16_enabled"] = neural_network.getUseFp16() if hasattr(neural_network, 'getUseFp16') else args.fp16
    else:
         metadata["nn_loaded"] = False

    metadata_filename = f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.json"
    metadata_path = os.path.join(args.output_dir, metadata_filename)
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata to {metadata_path}: {e}")

    print("Self-play finished.")


if __name__ == "__main__":
    args = parse_args()
    # Ensure the C++ module is loaded before potentially lengthy operations
    print("Checking C++ module availability...")
    if 'az' not in locals() or az is None:
         print("Fatal: C++ module (_alphazero_cpp) could not be loaded.")
         sys.exit(1)
    print("C++ module loaded successfully.")

    run_self_play(args)