"""
ELO rating system for evaluating AlphaZero model strength.
"""

import math
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np


def compute_expected_score(rating_a, rating_b):
    """
    Compute the expected score for player A against player B.
    
    Args:
        rating_a (float): ELO rating of player A
        rating_b (float): ELO rating of player B
        
    Returns:
        float: Expected score for player A (between 0 and 1)
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def calculate_elo_change(rating, opponent_rating, score, k_factor=32):
    """
    Calculate the change in ELO rating after a game.
    
    Args:
        rating (float): Current rating
        opponent_rating (float): Opponent's rating
        score (float): Actual score (1 for win, 0.5 for draw, 0 for loss)
        k_factor (float): K-factor for ELO calculation
        
    Returns:
        float: Change in rating
    """
    expected = compute_expected_score(rating, opponent_rating)
    return k_factor * (score - expected)


class EloRating:
    """
    ELO rating tracker for AlphaZero models.
    
    Args:
        initial_rating (float): Initial rating for new players
        k_factor (float): K-factor for ELO calculation
    """
    
    def __init__(self, initial_rating=1500.0, k_factor=32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = {}  # {player_id: rating}
        self.history = {}  # {player_id: [(timestamp, rating, opponent, result)]}
        
    def get_rating(self, player_id):
        """
        Get the current rating for a player.
        
        Args:
            player_id (str): Player identifier
            
        Returns:
            float: Current rating
        """
        return self.ratings.get(player_id, self.initial_rating)
    
    def add_game_result(self, player_id, opponent_id, score):
        """
        Add a game result and update ratings.
        
        Args:
            player_id (str): Player identifier
            opponent_id (str): Opponent identifier
            score (float): Score for player (1.0 for win, 0.5 for draw, 0.0 for loss)
            
        Returns:
            float: New rating for player
        """
        # Get current ratings
        player_rating = self.get_rating(player_id)
        opponent_rating = self.get_rating(opponent_id)
        
        # Calculate rating change
        rating_change = calculate_elo_change(
            player_rating, opponent_rating, score, self.k_factor
        )
        
        # Update ratings
        new_rating = player_rating + rating_change
        self.ratings[player_id] = new_rating
        
        # Update opponent rating
        opponent_score = 1.0 - score
        opponent_rating_change = calculate_elo_change(
            opponent_rating, player_rating, opponent_score, self.k_factor
        )
        new_opponent_rating = opponent_rating + opponent_rating_change
        self.ratings[opponent_id] = new_opponent_rating
        
        # Record history
        timestamp = datetime.datetime.now().isoformat()
        if player_id not in self.history:
            self.history[player_id] = []
        self.history[player_id].append((timestamp, new_rating, opponent_id, score))
        
        # Also record history for the opponent
        if opponent_id not in self.history:
            self.history[opponent_id] = []
        self.history[opponent_id].append((timestamp, new_opponent_rating, player_id, opponent_score))
        
        return new_rating
    
    def add_match_results(self, player_id, opponent_id, wins, draws, losses):
        """
        Add results from a match with multiple games.
        
        Args:
            player_id (str): Player identifier
            opponent_id (str): Opponent identifier
            wins (int): Number of wins
            draws (int): Number of draws
            losses (int): Number of losses
            
        Returns:
            tuple: (new_rating, rating_change)
        """
        initial_rating = self.get_rating(player_id)
        
        # Add individual game results
        for _ in range(wins):
            self.add_game_result(player_id, opponent_id, 1.0)
        
        for _ in range(draws):
            self.add_game_result(player_id, opponent_id, 0.5)
        
        for _ in range(losses):
            self.add_game_result(player_id, opponent_id, 0.0)
        
        # Return the final rating and total change
        final_rating = self.get_rating(player_id)
        return final_rating, final_rating - initial_rating
    
    def get_history(self, player_id):
        """
        Get rating history for a player.
        
        Args:
            player_id (str): Player identifier
            
        Returns:
            list: List of (timestamp, rating, opponent, result) tuples
        """
        return self.history.get(player_id, [])
    
    def save(self, filename):
        """
        Save ratings and history to a JSON file.
        
        Args:
            filename (str): File path to save to
        """
        data = {
            'initial_rating': self.initial_rating,
            'k_factor': self.k_factor,
            'ratings': self.ratings,
            'history': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename):
        """
        Load ratings and history from a JSON file.
        
        Args:
            filename (str): File path to load from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.initial_rating = data.get('initial_rating', 1500.0)
            self.k_factor = data.get('k_factor', 32.0)
            self.ratings = data.get('ratings', {})
            self.history = data.get('history', {})
            return True
        except Exception as e:
            print(f"Error loading ELO data: {e}")
            return False
    
    def print_ratings(self, top_n=None):
        """
        Print current ratings in descending order.
        
        Args:
            top_n (int, optional): Number of top players to show
        """
        sorted_ratings = sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        )
        
        if top_n:
            sorted_ratings = sorted_ratings[:top_n]
        
        print(f"ELO Ratings (k={self.k_factor}):")
        print("=" * 40)
        for player, rating in sorted_ratings:
            print(f"{player}: {rating:.1f}")