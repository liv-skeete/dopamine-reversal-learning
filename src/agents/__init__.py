"""
Dopamine Agents Module
=====================

Contains implementations of dopamine-related agent models:
- Prediction Error Agent
- Hedonic Agent  
- Incentive Salience Agent
"""

from .prediction_error_agent import PredictionErrorAgent
from .hedonic_agent import HedonicAgent
from .incentive_salience_agent import IncentiveSalienceAgent

__all__ = ['PredictionErrorAgent', 'HedonicAgent', 'IncentiveSalienceAgent']