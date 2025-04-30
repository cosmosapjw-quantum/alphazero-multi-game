"""
AlphaZero Multi-Game AI Engine

A reinforcement learning system that masters board games (Gomoku, Chess, Go)
through self-play without human knowledge.
"""

__version__ = '1.0.0'

# Import core components from C++ extension
try:
    from _alphazero_cpp import (
        # Enums
        GameType, GameResult, MCTSNodeSelection,
        
        # Game state
        IGameState, GomokuState, createGameState,
        
        # Neural network
        NeuralNetwork, createNeuralNetwork,
        
        # MCTS
        MCTSNode, TranspositionTable, ParallelMCTS,
        
        # Self-play
        MoveData, GameRecord, TrainingExample, Dataset, SelfPlayManager
    )
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension '_alphazero_cpp'. "
        "Make sure the extension is properly built and installed. "
        f"Original error: {e}"
    )

# Import Python components
from alphazero.models import DDWRandWireResNet
from alphazero.training import (
    AlphaZeroLoss, 
    WarmupCosineAnnealingLR,
    LinearWarmupScheduler,
    CyclicCosineAnnealingLR
)