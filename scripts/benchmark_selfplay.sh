#!/bin/bash
# Benchmark script to compare Python and C++ self-play performance

set -e

# Default settings
GAME="gomoku"
SIZE=9
NUM_GAMES=10
SIMULATIONS=200
THREADS=4
MODEL_PATH=""
OUTPUT_DIR="data/benchmark"
BATCH_SIZE=16
BATCH_TIMEOUT=10

# Create output directory
mkdir -p $OUTPUT_DIR

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --game)
      GAME="$2"
      shift 2
      ;;
    --size)
      SIZE="$2"
      shift 2
      ;;
    --num-games)
      NUM_GAMES="$2"
      shift 2
      ;;
    --simulations)
      SIMULATIONS="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-timeout)
      BATCH_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate model path
if [ -n "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTHON_OUTPUT_DIR="${OUTPUT_DIR}/python_${TIMESTAMP}"
CPP_OUTPUT_DIR="${OUTPUT_DIR}/cpp_${TIMESTAMP}"

mkdir -p "$PYTHON_OUTPUT_DIR"
mkdir -p "$CPP_OUTPUT_DIR"

# Function to run and time a command
run_benchmark() {
  local name=$1
  local cmd=$2
  local output_file=$3
  
  echo "Running $name benchmark..."
  echo "Command: $cmd"
  
  # Measure time with the 'time' command
  { time eval "$cmd" > "$output_file" 2>&1; } 2> "${output_file}.time"
  
  # Extract time information
  REAL_TIME=$(grep "real" "${output_file}.time" | awk '{print $2}')
  USER_TIME=$(grep "user" "${output_file}.time" | awk '{print $2}')
  SYS_TIME=$(grep "sys" "${output_file}.time" | awk '{print $2}')
  
  echo "$name completed in $REAL_TIME"
  echo "User time: $USER_TIME"
  echo "System time: $SYS_TIME"
  echo "---------------------------------"
}

# Check both implementations are available
if ! command -v python3 &> /dev/null; then
  echo "Python 3 not found, skipping Python benchmark"
  RUN_PYTHON=0
else
  RUN_PYTHON=1
fi

if [ ! -f "./build/src/selfplay/self_play" ]; then
  echo "C++ self_play executable not found, skipping C++ benchmark"
  RUN_CPP=0
else
  RUN_CPP=1
fi

# Prepare model arguments
if [ -n "$MODEL_PATH" ]; then
  MODEL_ARG="--model $MODEL_PATH"
else
  MODEL_ARG=""
fi

# Run Python benchmark
if [ $RUN_PYTHON -eq 1 ]; then
  PYTHON_CMD="python3 python/scripts/self_play.py $MODEL_ARG --game $GAME --size $SIZE --num-games $NUM_GAMES --simulations $SIMULATIONS --threads $THREADS --output-dir $PYTHON_OUTPUT_DIR --batch-size $BATCH_SIZE --batch-timeout $BATCH_TIMEOUT"
  run_benchmark "Python self-play" "$PYTHON_CMD" "${PYTHON_OUTPUT_DIR}/output.log"
fi

# Run C++ benchmark
if [ $RUN_CPP -eq 1 ]; then
  CPP_CMD="./build/src/selfplay/self_play $MODEL_ARG --game $GAME --size $SIZE --num-games $NUM_GAMES --simulations $SIMULATIONS --threads $THREADS --output-dir $CPP_OUTPUT_DIR --batch-size $BATCH_SIZE --batch-timeout $BATCH_TIMEOUT"
  run_benchmark "C++ self-play" "$CPP_CMD" "${CPP_OUTPUT_DIR}/output.log"
fi

# Compare results if both implementations were run
if [ $RUN_PYTHON -eq 1 ] && [ $RUN_CPP -eq 1 ]; then
  echo "Comparing results..."
  
  # Extract time information
  PYTHON_REAL=$(grep "real" "${PYTHON_OUTPUT_DIR}/output.log.time" | awk -F'm|s' '{print $1*60+$2}')
  CPP_REAL=$(grep "real" "${CPP_OUTPUT_DIR}/output.log.time" | awk -F'm|s' '{print $1*60+$2}')
  
  # Calculate speedup
  SPEEDUP=$(echo "scale=2; $PYTHON_REAL / $CPP_REAL" | bc)
  
  echo "Python execution time: $PYTHON_REAL seconds"
  echo "C++ execution time: $CPP_REAL seconds"
  echo "Speedup: ${SPEEDUP}x"
  
  # Extract more stats if available
  echo "Summary:"
  echo "---------------------------------"
  
  # Extract Python stats
  PYTHON_MOVES=$(grep "Total moves:" "${PYTHON_OUTPUT_DIR}/output.log" | awk '{print $3}')
  PYTHON_MOVES_PER_SEC=$(grep "Avg moves/second:" "${PYTHON_OUTPUT_DIR}/output.log" | awk '{print $3}')
  
  # Extract C++ stats
  CPP_MOVES=$(grep "Total moves:" "${CPP_OUTPUT_DIR}/output.log" | awk '{print $3}')
  CPP_MOVES_PER_SEC=$(grep "Average moves per second:" "${CPP_OUTPUT_DIR}/output.log" | awk '{print $5}')
  
  if [ -n "$PYTHON_MOVES" ] && [ -n "$CPP_MOVES" ]; then
    echo "Python total moves: $PYTHON_MOVES"
    echo "C++ total moves: $CPP_MOVES"
  fi
  
  if [ -n "$PYTHON_MOVES_PER_SEC" ] && [ -n "$CPP_MOVES_PER_SEC" ]; then
    echo "Python moves per second: $PYTHON_MOVES_PER_SEC"
    echo "C++ moves per second: $CPP_MOVES_PER_SEC"
  fi
  
  # Save summary to file
  {
    echo "Benchmark Summary ($TIMESTAMP)"
    echo "---------------------------------"
    echo "Game: $GAME"
    echo "Board size: $SIZE"
    echo "Number of games: $NUM_GAMES"
    echo "Simulations per move: $SIMULATIONS"
    echo "Threads: $THREADS"
    echo "Batch size: $BATCH_SIZE"
    echo "Batch timeout: $BATCH_TIMEOUT"
    echo "---------------------------------"
    echo "Python execution time: $PYTHON_REAL seconds"
    echo "C++ execution time: $CPP_REAL seconds"
    echo "Speedup: ${SPEEDUP}x"
    
    if [ -n "$PYTHON_MOVES" ] && [ -n "$CPP_MOVES" ]; then
      echo "Python total moves: $PYTHON_MOVES"
      echo "C++ total moves: $CPP_MOVES"
    fi
    
    if [ -n "$PYTHON_MOVES_PER_SEC" ] && [ -n "$CPP_MOVES_PER_SEC" ]; then
      echo "Python moves per second: $PYTHON_MOVES_PER_SEC"
      echo "C++ moves per second: $CPP_MOVES_PER_SEC"
      
      # Calculate moves per second speedup
      MOVES_SPEEDUP=$(echo "scale=2; $CPP_MOVES_PER_SEC / $PYTHON_MOVES_PER_SEC" | bc)
      echo "Moves per second speedup: ${MOVES_SPEEDUP}x"
    fi
  } > "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
  
  echo "Summary saved to ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
fi

echo "Benchmark completed!"