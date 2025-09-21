"""
Incentive Salience Agent
========================

Implements the incentive salience agent from the original Colab code.
Models motivational salience and cue reactivity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from src.utils.helpers import get_object_index, get_outcome_index, softmax


class IncentiveSalienceAgent:
    """
    Incentive salience agent with enhanced motivational salience.
    
    This agent models the incentive salience aspect of dopamine function
    with enhanced motivational value for specific cues.
    """
    
    def __init__(self, group: str, bias: float, v2q_thresh: float = 0.6, 
                 switch_window: int = 5):
        """
        Initialize the incentive salience agent.
        
        Args:
            group: Agent group ("Normal" or "Addicted")
            bias: Salience bias parameter
            v2q_thresh: Threshold for switching from V to Q learning
            switch_window: Window size for meta-policy switching
        """
        self.group = group
        self.bias = bias
        self.Q = np.zeros(3)  # Q-values for objects
        self.V = np.zeros(18)  # V-values for outcome states
        self.alpha_Q = 0.05
        self.alpha_V = 0.3
        self.salience_bonus = bias if group == "Addicted" else 0.0
        self.temp_Q = 1.0
        self.temp_V = 0.7
        self.meta_mode = "V"
        self.v_recent = []
        self.v2q_thresh = v2q_thresh
        self.switch_window = switch_window
        
        # History tracking
        self.choice_history = []
        self.reward_history = []
        self.mode_history = []
        self.salience_applied = []
    
    def v_switch_prob(self) -> float:
        """
        Get probability of using V-learning.
        
        Returns:
            Probability (1.0 if using V, 0.0 otherwise)
        """
        return 1.0 if self.meta_mode == "V" else 0.0
    
    def choose(self, objs: List[np.ndarray], correct_obj: np.ndarray) -> Tuple[int, bool]:
        """
        Choose an object with incentive salience modulation.
        
        Args:
            objs: List of object vectors
            correct_obj: Correct object vector
            
        Returns:
            Tuple of (choice index, whether V-learning was used)
        """
        indices = [get_object_index(obj) for obj in objs]
        
        # Apply salience bonus to object 0 for addicted group
        Qvals = self.Q[indices].copy()
        for i, obj_id in enumerate(indices):
            if obj_id == 0:  # Only obj0 gets salience bonus
                Qvals[i] += self.salience_bonus
        
        Q_probs = softmax(Qvals, self.temp_Q)
        
        if self.meta_mode == "V":
            future_vals = []
            for i, obj in enumerate(objs):
                obj_id = get_object_index(obj)
                loc = i
                rew = int(np.array_equal(obj, correct_obj))
                state_idx = get_outcome_index(loc, obj_id, rew)
                v_val = self.V[state_idx]
                
                # Apply salience bonus to object 0 for V-values
                if obj_id == 0:
                    v_val += self.salience_bonus
                
                future_vals.append(v_val)
            
            V_probs = softmax(np.array(future_vals), self.temp_V)
            mixed_probs = 0.5 * Q_probs + 0.5 * V_probs  # Even split
        else:
            mixed_probs = Q_probs
        
        choice = np.random.choice(len(objs), p=mixed_probs)
        used_v = (self.meta_mode == "V")
        
        # Record choice and salience application
        self.choice_history.append(choice)
        self.mode_history.append(used_v)
        self.salience_applied.append(self.salience_bonus > 0)
        
        return choice, used_v
    
    def update(self, choice_idx: int, objs: List[np.ndarray], 
               reward: int, correct_obj: np.ndarray):
        """
        Update value estimates.
        
        Args:
            choice_idx: Index of chosen object
            objs: List of object vectors
            reward: Received reward
            correct_obj: Correct object vector
        """
        obj_id = get_object_index(objs[choice_idx])
        
        # V-learning update
        if self.meta_mode == "V":
            loc = choice_idx
            rew = int(np.array_equal(objs[choice_idx], correct_obj))
            state_idx = get_outcome_index(loc, obj_id, rew)
            delta_v = reward - self.V[state_idx]
            self.V[state_idx] += self.alpha_V * delta_v
        
        # Q-learning update
        delta_q = reward - self.Q[obj_id]
        self.Q[obj_id] += self.alpha_Q * delta_q
        
        # Record reward
        self.reward_history.append(reward)
    
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
            'modes': self.mode_history.copy(),
            'salience_applied': self.salience_applied.copy(),
            'Q_values': self.Q.copy(),
            'V_values': self.V.copy()
        }
    
    def reset_history(self):
        """Reset agent's history while keeping parameters."""
        self.choice_history = []
        self.reward_history = []
        self.mode_history = []
        self.salience_applied = []