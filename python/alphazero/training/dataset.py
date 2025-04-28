"""
Dataset utilities for AlphaZero training.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyalphazero as az


class AlphaZeroDataset(Dataset):
    """
    PyTorch Dataset for AlphaZero training data.
    
    Loads and processes game records generated during self-play.
    
    Args:
        examples (list): List of TrainingExample objects
        transform (callable, optional): Optional transform to apply to samples
    """
    
    def __init__(self, examples, transform=None):
        self.examples = examples
        self.transform = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to PyTorch tensors
        state = torch.FloatTensor(example.state)
        policy = torch.FloatTensor(example.policy)
        value = torch.FloatTensor([example.value])
        
        if self.transform:
            state, policy, value = self.transform(state, policy, value)
        
        return state, policy, value


class GameDatasetBuilder:
    """
    Utility for building datasets from self-play game records.
    
    Args:
        game_type (GameType): Type of game
        use_enhanced_features (bool): Whether to use enhanced tensor features
        include_augmentations (bool): Whether to include data augmentations
    """
    
    def __init__(self, game_type, use_enhanced_features=True, include_augmentations=True):
        self.game_type = game_type
        self.use_enhanced_features = use_enhanced_features
        self.include_augmentations = include_augmentations
        self.dataset = az.Dataset()
    
    def add_game_record(self, record):
        """
        Add a game record to the dataset.
        
        Args:
            record (GameRecord): Game record to add
        """
        self.dataset.addGameRecord(record, self.use_enhanced_features)
        
    def add_games_from_directory(self, directory):
        """
        Add all game records from a directory.
        
        Args:
            directory (str): Directory containing game record files
            
        Returns:
            int: Number of games loaded
        """
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                try:
                    path = os.path.join(directory, filename)
                    record = az.GameRecord.loadFromFile(path)
                    self.dataset.addGameRecord(record, self.use_enhanced_features)
                    count += 1
                except Exception as e:
                    print(f"Error loading game {filename}: {e}")
        return count
    
    def extract_examples(self):
        """
        Extract training examples from the game records.
        
        Returns:
            list: List of TrainingExample objects
        """
        self.dataset.extractExamples(self.include_augmentations)
        return self.dataset.getRandomSubset(self.dataset.size())
    
    def build_torch_dataset(self):
        """
        Build a PyTorch dataset from the game records.
        
        Returns:
            AlphaZeroDataset: PyTorch dataset
        """
        examples = self.extract_examples()
        return AlphaZeroDataset(examples)
    
    def create_data_loader(self, batch_size=128, shuffle=True, num_workers=4):
        """
        Create a PyTorch DataLoader from the game records.
        
        Args:
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker threads
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        dataset = self.build_torch_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )