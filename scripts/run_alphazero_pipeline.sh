#!/bin/bash

# AlphaZero Pipeline for Gomoku
# This script automates the full AlphaZero training pipeline:
# 1. Generate self-play data
# 2. Train the neural network
# 3. Evaluate the new model

set -e  # Exit immediately if a command exits with a non-zero status

# Default parameters
BUILD_DIR="../build"
NUM_ITERATIONS=10
NUM_GAMES=100
NUM_WORKERS=4
BOARD_SIZE=15
SIMULATIONS=800
THREADS=2
USE_RENJU=false
USE_GPU=false
TRAIN_EPOCHS=20
EVALUATION_GAMES=20
WIN_THRESHOLD=0.55  # 55% win rate required to accept new model

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --build-dir)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        --iterations)
            NUM_ITERATIONS="$2"
            shift
            shift
            ;;
        --games)
            NUM_GAMES="$2"
            shift
            shift
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift
            shift
            ;;
        --board-size)
            BOARD_SIZE="$2"
            shift
            shift
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift
            shift
            ;;
        --threads)
            THREADS="$2"
            shift
            shift
            ;;
        --train-epochs)
            TRAIN_EPOCHS="$2"
            shift
            shift
            ;;
        --evaluation-games)
            EVALUATION_GAMES="$2"
            shift
            shift
            ;;
        --win-threshold)
            WIN_THRESHOLD="$2"
            shift
            shift
            ;;
        --use-renju)
            USE_RENJU=true
            shift
            ;;
        --use-gpu)
            USE_GPU=true
            shift
            ;;
        --help)
            echo "AlphaZero Pipeline for Gomoku"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build-dir DIR          Build directory (default: ../build)"
            echo "  --iterations N           Number of iterations (default: 10)"
            echo "  --games N                Number of self-play games per iteration (default: 100)"
            echo "  --workers N              Number of parallel workers (default: 4)"
            echo "  --board-size N           Board size (default: 15)"
            echo "  --simulations N          MCTS simulations per move (default: 800)"
            echo "  --threads N              Threads per game (default: 2)"
            echo "  --train-epochs N         Training epochs per iteration (default: 20)"
            echo "  --evaluation-games N     Number of evaluation games (default: 20)"
            echo "  --win-threshold X        Win rate threshold for accepting new model (default: 0.55)"
            echo "  --use-renju              Use Renju rules"
            echo "  --use-gpu                Use GPU for neural network"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Executable path
CLI_PATH="${BUILD_DIR}/bin/alphazero_cli"

# Check if executable exists
if [ ! -f "$CLI_PATH" ]; then
    echo "Error: AlphaZero executable not found at $CLI_PATH"
    echo "Make sure to build the project first"
    exit 1
fi

# Create directories
mkdir -p data
mkdir -p models

# Initial model (random network)
CURRENT_MODEL=""
BEST_MODEL=""

echo "Starting AlphaZero pipeline for Gomoku"
echo "======================================"
echo "Board size: $BOARD_SIZE"
echo "Simulations: $SIMULATIONS"
echo "Threads per game: $THREADS"
echo "Rules: $([ "$USE_RENJU" = true ] && echo "Renju" || echo "Standard Gomoku")"
echo "GPU: $([ "$USE_GPU" = true ] && echo "Enabled" || echo "Disabled")"
echo "======================================="

# Main training loop
for ((i=1; i<=$NUM_ITERATIONS; i++))
do
    echo
    echo "Iteration $i/$NUM_ITERATIONS"
    echo "-----------------"
    
    # 1. Generate self-play data
    ITERATION_DIR="data/iteration_$i"
    mkdir -p "$ITERATION_DIR"
    
    echo "Generating $NUM_GAMES self-play games..."
    
    # Prepare model argument if we have a model
    MODEL_ARG=""
    if [ -n "$CURRENT_MODEL" ]; then
        MODEL_ARG="--model $CURRENT_MODEL"
    fi
    
    # Prepare GPU argument if requested
    GPU_ARG=""
    if [ "$USE_GPU" = true ]; then
        GPU_ARG="--use-gpu"
    fi
    
    # Prepare Renju argument if requested
    RENJU_ARG=""
    if [ "$USE_RENJU" = true ]; then
        RENJU_ARG="--use-renju"
    fi
    
    # Python script for parallel self-play
    python3 python/alphazero_self_play.py \
        --executable "$CLI_PATH" \
        --output-dir "$ITERATION_DIR" \
        --games "$NUM_GAMES" \
        --workers "$NUM_WORKERS" \
        --board-size "$BOARD_SIZE" \
        --simulations "$SIMULATIONS" \
        --threads "$THREADS" \
        $MODEL_ARG $GPU_ARG $RENJU_ARG
    
    # 2. Train neural network
    echo "Training neural network on self-play data..."
    
    NEW_MODEL="models/gomoku_iteration_${i}.pt"
    
    # Optional previous model argument
    CHECKPOINT_ARG=""
    if [ -n "$CURRENT_MODEL" ]; then
        CHECKPOINT_ARG="--checkpoint $CURRENT_MODEL"
    fi
    
    python3 python/alphazero_train.py \
        --data-dir "$ITERATION_DIR" \
        --output-dir "models" \
        --game "gomoku" \
        --board-size "$BOARD_SIZE" \
        --epochs "$TRAIN_EPOCHS" \
        $CHECKPOINT_ARG
    
    # Use the exported model from training
    NEW_MODEL="models/gomoku_final.pt"
    
    # 3. Evaluate new model against current best
    if [ -n "$CURRENT_MODEL" ]; then
        echo "Evaluating new model against current best..."
        
        # Create a directory for evaluation games
        EVAL_DIR="data/evaluation_${i}"
        mkdir -p "$EVAL_DIR"
        
        # Track wins
        NEW_MODEL_WINS=0
        CURRENT_MODEL_WINS=0
        DRAWS=0
        
        # Play evaluation games
        for ((game=1; game<=$EVALUATION_GAMES; game++))
        do
            echo "Evaluation game $game/$EVALUATION_GAMES"
            
            # Decide which model plays first (alternate)
            if [ $((game % 2)) -eq 1 ]; then
                # New model plays first
                MODEL1="$NEW_MODEL"
                MODEL2="$CURRENT_MODEL"
                MODEL1_NAME="New"
                MODEL2_NAME="Current"
            else
                # Current model plays first
                MODEL1="$CURRENT_MODEL"
                MODEL2="$NEW_MODEL"
                MODEL1_NAME="Current"
                MODEL2_NAME="New"
            fi
            
            # Play evaluation game
            EVAL_GAME="${EVAL_DIR}/game_${game}.json"
            
            # First half of the game with MODEL1
            echo "  $MODEL1_NAME model playing as Black..."
            "$CLI_PATH" \
                --game gomoku \
                --board-size "$BOARD_SIZE" \
                --simulations "$SIMULATIONS" \
                --threads "$THREADS" \
                --model "$MODEL1" \
                $GPU_ARG $RENJU_ARG \
                --selfplay \
                --output "$EVAL_DIR/temp_game.json" \
                --temperature-init 0.1 \
                --temperature-final 0.1 \
                --temperature-threshold 0
            
            # Use the state from the last move
            LAST_STATE=$(grep -o '"action": [0-9]*' "$EVAL_DIR/temp_game.json" | tail -1)
            
            # Second half of the game with MODEL2
            echo "  $MODEL2_NAME model playing as White..."
            "$CLI_PATH" \
                --game gomoku \
                --board-size "$BOARD_SIZE" \
                --simulations "$SIMULATIONS" \
                --threads "$THREADS" \
                --model "$MODEL2" \
                $GPU_ARG $RENJU_ARG \
                --selfplay \
                --output "$EVAL_GAME" \
                --temperature-init 0.1 \
                --temperature-final 0.1 \
                --temperature-threshold 0
            
            # Determine the winner
            RESULT=$(grep -o '"result": [0-9]' "$EVAL_GAME" | cut -d' ' -f2)
            
            if [ "$RESULT" = "1" ]; then
                if [ $((game % 2)) -eq 1 ]; then
                    echo "  New model won as Black"
                    NEW_MODEL_WINS=$((NEW_MODEL_WINS + 1))
                else
                    echo "  Current model won as Black"
                    CURRENT_MODEL_WINS=$((CURRENT_MODEL_WINS + 1))
                fi
            elif [ "$RESULT" = "2" ]; then
                if [ $((game % 2)) -eq 1 ]; then
                    echo "  Current model won as White"
                    CURRENT_MODEL_WINS=$((CURRENT_MODEL_WINS + 1))
                else
                    echo "  New model won as White"
                    NEW_MODEL_WINS=$((NEW_MODEL_WINS + 1))
                fi
            else
                echo "  Game ended in a draw"
                DRAWS=$((DRAWS + 1))
            fi
            
            echo "  Score: New model: $NEW_MODEL_WINS, Current model: $CURRENT_MODEL_WINS, Draws: $DRAWS"
        done
        
        # Calculate win rate
        TOTAL_GAMES=$((NEW_MODEL_WINS + CURRENT_MODEL_WINS + DRAWS))
        WIN_RATE=$(echo "scale=3; $NEW_MODEL_WINS / $TOTAL_GAMES" | bc)
        echo "Evaluation results:"
        echo "  New model wins: $NEW_MODEL_WINS"
        echo "  Current model wins: $CURRENT_MODEL_WINS"
        echo "  Draws: $DRAWS"
        echo "  Win rate: $WIN_RATE (threshold: $WIN_THRESHOLD)"
        
        # Decide whether to keep the new model
        if (( $(echo "$WIN_RATE >= $WIN_THRESHOLD" | bc -l) )); then
            echo "New model exceeds win threshold, accepting as best model"
            CURRENT_MODEL="$NEW_MODEL"
            BEST_MODEL="$NEW_MODEL"
            cp "$NEW_MODEL" "models/gomoku_best.pt"
        else
            echo "New model below win threshold, keeping current model"
        fi
    else
        # First iteration, no evaluation needed
        echo "First iteration, accepting model without evaluation"
        CURRENT_MODEL="$NEW_MODEL"
        BEST_MODEL="$NEW_MODEL"
        cp "$NEW_MODEL" "models/gomoku_best.pt"
    fi
    
    echo "Iteration $i completed"
    echo "Current best model: $BEST_MODEL"
    echo
done

echo "AlphaZero pipeline completed"
echo "Final best model: $BEST_MODEL"