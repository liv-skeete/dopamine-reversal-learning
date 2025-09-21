"""
Helper Functions
================

Common utility functions used across dopamine research agents and experiments.
"""

import numpy as np
from typing import List, Tuple

# Constants
NUM_OBJECTS = 3
OBJECT_IDS = [0, 1, 2]
LOCATIONS = [0, 1, 2]
OUTCOME_STATES = [(loc, obj, rew) for loc in LOCATIONS for obj in OBJECT_IDS for rew in [0, 1]]
GROUP_ORDER = ["Normal", "Addicted"]

def get_object_index(obj_vector: np.ndarray) -> int:
    """
    Get the index of an object from its one-hot encoded vector.
    
    Args:
        obj_vector: One-hot encoded object vector
        
    Returns:
        Object index (0, 1, or 2)
    """
    return np.argmax(obj_vector)

def get_outcome_index(location: int, obj_id: int, reward: int) -> int:
    """
    Get the index of an outcome state.
    
    Args:
        location: Location index (0, 1, or 2)
        obj_id: Object index (0, 1, or 2)
        reward: Reward value (0 or 1)
        
    Returns:
        Outcome state index
    """
    return OUTCOME_STATES.index((location, obj_id, reward))

def describe_state(state_tuple: Tuple[int, int, int]) -> str:
    """
    Create a human-readable description of a state.
    
    Args:
        state_tuple: (location, object, reward) tuple
        
    Returns:
        String description of the state
    """
    loc, obj, rew = state_tuple
    loc_str = ["L", "C", "R"][loc]
    rew_str = "R" if rew == 1 else "Ø"
    return f"{loc_str}-{rew_str}-obj{obj}"

def softmax(qvals: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Compute softmax probabilities.
    
    Args:
        qvals: Array of Q-values
        temp: Temperature parameter
        
    Returns:
        Softmax probabilities
    """
    logits = qvals / temp
    exp_q = np.exp(logits - np.max(logits))
    return exp_q / np.sum(exp_q)

def create_object_vectors() -> List[np.ndarray]:
    """
    Create one-hot encoded object vectors.
    
    Returns:
        List of object vectors
    """
    return [np.eye(NUM_OBJECTS)[i] for i in range(NUM_OBJECTS)]