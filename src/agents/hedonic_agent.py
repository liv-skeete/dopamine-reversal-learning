"""
Hedonic Agent
=============

Implements the hedonic agent from the original Colab code.
Models reward sensitivity and hedonic responses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from src.utils.helpers import get_object_index, get_outcome_index, softmax


class HedonicAgent:
    """
    Hedonic agent with enhanced reward sensitivity.
    
    This agent models the hedonic aspect of dopamine function with
    enhanced reward processing for addicted individuals.
    """
    
    def __init__(self, group: str, bias: float, v2q_thresh: float = 0.6, 
                 switch_window: int = 5):
        """
        Initialize the hedonic agent.
        
        Args:
            group: Agent group ("Normal" or "Addicted")
            bias: Reward bias parameter
            v2q_thresh: Threshold for switching from V to Q learning
            switch_window: Window size for meta-policy switching
        """
        self.group = group
        self.bias = bias
        self.Q = np.zeros(3)  # Q-values for objects
        self.V = np.zeros(18)  # V-values for outcome states
        self.alpha_Q = 0.05
        self.alpha_V = 0.3
        self.temp_Q = 1.0
        self.temp_V = 0.7
        self.allow_V = True
        self.session = 0
        self.trial = 0
        self.meta_mode = "V"
        self.v_recent = []
        self.v2q_thresh = v2q_thresh
        self.switch_window = switch_window
        
        # History tracking
        self.choice_history = []
        self.reward_history = []
        self.mode_history = []
        self.modified_rewards = []
    
    def v_switch_prob(self) -> float:
        """
        Get probability of using V-learning.
        
        Returns:
            Probability (1.0 if using V, 0.0 otherwise)
        """
        return 1.0 if self.meta_mode == "V" else 0.0
    
    def choose(self, objs: List[np.ndarray], correct_obj: np.ndarray) -> Tuple[int, bool]:
        """
        Choose an object based on current policy.
        
        Args:
            objs: List of object vectors
            correct_obj: Correct object vector
            
        Returns:
            Tuple of (choice index, whether V-learning was used)
        """
        indices = [get_object_index(obj) for obj in objs]
        Q_probs = softmax(self.Q[indices], self.temp_Q)
        
        if self.meta_mode == "V":
            future_vals = []
            for i, obj in enumerate(objs):
                obj_id = get_object_index(obj)
                loc = i
                rew = int(np.array_equal(obj, correct_obj))
                state_idx = get_outcome_index(loc, obj_id, rew)
                future_vals.append(self.V[state_idx])
            
            V_probs = softmax(np.array(future_vals), self.temp_V)
            mixed_probs = 0.5 * Q_probs + 0.5 * V_probs  # Even split
        else:
            mixed_probs = Q_probs
        
        choice = np.random.choice(len(objs), p=mixed_probs)
        used_v = (self.meta_mode == "V")
        
        # Record choice
        self.choice_history.append(choice)
        self.mode_history.append(used_v)
        
        return choice, used_v
    
    def update(self, choice_idx: int, objs: List[np.ndarray], 
               reward: int, correct_obj: np.ndarray):
        """
        Update value estimates with hedonic modulation.
        
        Args:
            choice_idx: Index of chosen object
            objs: List of object vectors
            reward: Received reward
            correct_obj: Correct object vector
        """
        obj_id = get_object_index(objs[choice_idx])
        
        # Hedonic modulation for addicted group
        mod_reward = reward
        if self.bias > 0 and self.group == "Addicted":
            if reward == 1:
                mod_reward = 1.0 + self.bias  # Enhanced reward
            else:
                mod_reward = 0.1 * self.bias  # Reduced punishment
        
        # V-learning update
        if self.meta_mode == "V":
            loc = choice_idx
            rew = int(np.array_equal(objs[choice_idx], correct_obj))
            state_idx = get_outcome_index(loc, obj_id, rew)
            delta_v = mod_reward - self.V[state_idx]
            self.V[state_idx] += self.alpha_V * delta_v
        
        # Q-learning update
        delta_q = mod_reward - self.Q[obj_id]
        self.Q[obj_id] += self.alpha_Q * delta_q
        
        # Record rewards
        self.reward_history.append(reward)
        self.modified_rewards.append(mod_reward)
    
    def reset_session(self):
        """
        Reset agent state for a new session.
        """
        self.trial = 0
        self.meta_mode = "V"
        self.v_recent = []
    
    def update_meta_policy(self, v_correct: int):
        """
        Update meta-policy based on V-learning performance.
        
        Args:
            v_correct: Whether V-learning choice was correct
        """
        self.v_recent.append(v_correct)
        if len(self.v_recent) > self.switch_window:
            self.v_recent.pop(0)
        
        # If in V mode, switch to Q if mean V acc goes below threshold
        if self.meta_mode == "V" and len(self.v_recent) == self.switch_window:
            avg_acc = np.mean(self.v_recent)
            if avg_acc < self.v2q_thresh:
                self.meta_mode = "Q"
    
    def get_history(self) -> Dict[str, List]:
        """
        Get agent's learning history.
        
        Returns:
            Dictionary with choice, reward, and mode history
        """
        return {
            'choices': self.choice_history.copy(),
            'rewards': self.reward_history.copy(),
            'modified_rewards': self.modified_rewards.copy(),
            'modes': self.mode_history.copy(),
            'Q_values': self.Q.copy(),
            'V_values': self.V.copy()
        }
    
    def reset_history(self):
        """Reset agent's history while keeping parameters."""
        self.choice_history = []
        self.reward_history = []
        self.modified_rewards = []
        self.mode_history = []