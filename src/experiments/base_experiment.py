"""
Base Experiment Framework
========================

Provides a foundation for designing and running dopamine-related experiments.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import os


class BaseExperiment:
    """Base class for all dopamine research experiments."""
    
    def __init__(self, experiment_name: str, output_dir: str = "./results"):
        """
        Initialize the base experiment.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.trials = []
        self.parameters = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def setup(self, **kwargs):
        """
        Setup the experiment with parameters.
        
        Args:
            **kwargs: Experiment-specific parameters
        """
        self.parameters.update(kwargs)
        self.start_time = datetime.now()
        
    def run_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single trial of the experiment.
        
        Args:
            trial_data: Data for this trial
            
        Returns:
            Trial results
        """
        raise NotImplementedError("Subclasses must implement run_trial")
    
    def run(self, num_trials: int = 100, **trial_kwargs) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Args:
            num_trials: Number of trials to run
            **trial_kwargs: Additional trial parameters
            
        Returns:
            Experiment results
        """
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Parameters: {self.parameters}")
        
        for trial_idx in range(num_trials):
            trial_data = {
                'trial_id': trial_idx,
                'timestamp': datetime.now().isoformat(),
                **trial_kwargs
            }
            
            trial_result = self.run_trial(trial_data)
            self.trials.append({
                **trial_data,
                **trial_result
            })
            
            if trial_idx % 10 == 0:
                print(f"Completed trial {trial_idx}/{num_trials}")
        
        self.end_time = datetime.now()
        self._analyze_results()
        self._save_results()
        
        return self.results
    
    def _analyze_results(self):
        """Analyze and summarize experiment results."""
        # Basic analysis - can be overridden by subclasses
        self.results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            'num_trials': len(self.trials),
            'parameters': self.parameters,
            'summary_stats': self._compute_summary_stats()
        }
    
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from trials."""
        if not self.trials:
            return {}
            
        # Extract numeric results for statistical analysis
        numeric_results = {}
        for key in self.trials[0].keys():
            if isinstance(self.trials[0][key], (int, float)):
                values = [trial[key] for trial in self.trials]
                numeric_results[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return numeric_results
    
    def _save_results(self):
        """Save experiment results to files."""
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'summary': self.results,
                'trials': self.trials
            }, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    
    def get_progress(self) -> float:
        """Get experiment progress as a percentage."""
        if not self.trials or not self.parameters.get('num_trials'):
            return 0.0
        return len(self.trials) / self.parameters.get('num_trials', 1) * 100
    
    def reset(self):
        """Reset the experiment while keeping parameters."""
        self.trials = []
        self.results = {}
        self.start_time = None
        self.end_time = None


class PredictionErrorExperiment(BaseExperiment):
    """Experiment framework for prediction error studies."""
    
    def __init__(self, agent, **kwargs):
        super().__init__("prediction_error", **kwargs)
        self.agent = agent
    
    def run_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        # Example prediction error trial
        state = f"state_{np.random.randint(10)}"
        reward = np.random.normal(1.0, 0.5)
        next_state = f"state_{np.random.randint(10)}"
        
        prediction_error = self.agent.compute_prediction_error(state, reward, next_state)
        self.agent.update_value_estimate(state, prediction_error)
        
        return {
            'state': state,
            'reward': reward,
            'next_state': next_state,
            'prediction_error': prediction_error,
            'value_estimate': self.agent.value_estimates.get(state, 0)
        }


class HedonicExperiment(BaseExperiment):
    """Experiment framework for hedonic response studies."""
    
    def __init__(self, agent, **kwargs):
        super().__init__("hedonic", **kwargs)
        self.agent = agent
    
    def run_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        # Example hedonic trial
        stimulus = f"stimulus_{np.random.randint(5)}"
        intensity = np.random.uniform(0.5, 2.0)
        novelty = np.random.uniform(0.1, 1.0)
        
        hedonic_response = self.agent.compute_hedonic_response(stimulus, intensity, novelty)
        
        return {
            'stimulus': stimulus,
            'intensity': intensity,
            'novelty': novelty,
            'hedonic_response': hedonic_response
        }


class IncentiveSalienceExperiment(BaseExperiment):
    """Experiment framework for incentive salience studies."""
    
    def __init__(self, agent, **kwargs):
        super().__init__("incentive_salience", **kwargs)
        self.agent = agent
    
    def run_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        # Example incentive salience trial
        stimulus = f"cue_{np.random.randint(5)}"
        reward_value = np.random.uniform(0.1, 1.0)
        
        incentive_value = self.agent.compute_incentive_value(stimulus, reward_value)
        motivation = self.agent.get_motivation_level(stimulus)
        should_act = self.agent.should_act(stimulus)
        
        # Simulate action outcome
        action_success = np.random.random() > 0.3  # 70% success rate
        self.agent.record_action(stimulus, should_act, action_success)
        self.agent.decay_incentives()
        
        return {
            'stimulus': stimulus,
            'reward_value': reward_value,
            'incentive_value': incentive_value,
            'motivation': motivation,
            'should_act': should_act,
            'action_success': action_success if should_act else None
        }