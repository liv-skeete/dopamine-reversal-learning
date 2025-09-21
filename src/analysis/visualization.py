"""
Visualization Module
===================

Data visualization tools for dopamine research experiments.
Provides plotting functions for different types of experimental results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import os


class DopamineVisualizer:
    """Visualization tools for dopamine research data."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = (10, 6)
    
    def plot_prediction_errors(self, prediction_errors: List[float], 
                             title: str = "Prediction Errors Over Time",
                             save_path: Optional[str] = None):
        """
        Plot prediction errors over time.
        
        Args:
            prediction_errors: List of prediction error values
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        plt.plot(prediction_errors, 'b-', alpha=0.7, linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Trial')
        plt.ylabel('Prediction Error')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hedonic_responses(self, responses: List[float], 
                             stimulus_types: Optional[List[str]] = None,
                             title: str = "Hedonic Responses",
                             save_path: Optional[str] = None):
        """
        Plot hedonic responses.
        
        Args:
            responses: List of hedonic response values
            stimulus_types: List of stimulus types for coloring (optional)
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        if stimulus_types and len(stimulus_types) == len(responses):
            unique_stimuli = list(set(stimulus_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stimuli)))
            
            for i, stimulus in enumerate(unique_stimuli):
                indices = [j for j, s in enumerate(stimulus_types) if s == stimulus]
                plt.scatter(indices, [responses[j] for j in indices], 
                           color=colors[i], label=stimulus, alpha=0.7)
            plt.legend()
        else:
            plt.plot(responses, 'g-', alpha=0.7, linewidth=2)
        
        plt.xlabel('Trial')
        plt.ylabel('Hedonic Response')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_motivation_trend(self, motivation_history: List[float],
                            title: str = "Motivation Over Time",
                            save_path: Optional[str] = None):
        """
        Plot motivation levels over time.
        
        Args:
            motivation_history: List of motivation values
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        plt.plot(motivation_history, 'purple', alpha=0.7, linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Motivation Level')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparative_analysis(self, data_dict: Dict[str, List[float]],
                                title: str = "Comparative Analysis",
                                xlabel: str = "Condition",
                                ylabel: str = "Value",
                                save_path: Optional[str] = None):
        """
        Plot comparative analysis of different conditions.
        
        Args:
            data_dict: Dictionary of condition names to value lists
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        conditions = list(data_dict.keys())
        values = [np.mean(data_dict[cond]) for cond in conditions]
        errors = [np.std(data_dict[cond]) for cond in conditions]
        
        bars = plt.bar(conditions, values, yerr=errors, 
                      capsize=5, alpha=0.7, color='skyblue')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              save_path: Optional[str] = None):
        """
        Plot correlation matrix for experimental data.
        
        Args:
            data: Pandas DataFrame with experimental data
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        
        # Compute correlation matrix
        corr = data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_experiment_dashboard(self, experiment_results: Dict[str, Any],
                                  save_dir: Optional[str] = None):
        """
        Create a comprehensive dashboard for experiment results.
        
        Args:
            experiment_results: Dictionary containing experiment results
            save_dir: Directory to save dashboard plots (optional)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Extract data from results
        trials = experiment_results.get('trials', [])
        if not trials:
            print("No trial data available for dashboard")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trials)
        
        # Create multiple plots
        if 'prediction_error' in df.columns:
            self.plot_prediction_errors(
                df['prediction_error'].tolist(),
                "Prediction Errors Over Trials",
                os.path.join(save_dir, 'prediction_errors.png') if save_dir else None
            )
        
        if 'hedonic_response' in df.columns:
            stimulus_types = df.get('stimulus', None)
            self.plot_hedonic_responses(
                df['hedonic_response'].tolist(),
                stimulus_types.tolist() if stimulus_types is not None else None,
                "Hedonic Responses",
                os.path.join(save_dir, 'hedonic_responses.png') if save_dir else None
            )
        
        if 'motivation' in df.columns:
            self.plot_motivation_trend(
                df['motivation'].tolist(),
                "Motivation Over Time",
                os.path.join(save_dir, 'motivation.png') if save_dir else None
            )
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            self.plot_correlation_matrix(
                df[numeric_cols],
                "Feature Correlations",
                os.path.join(save_dir, 'correlations.png') if save_dir else None
            )


# Utility function for quick plotting
def quick_plot(data: List[float], title: str = "Plot", **kwargs):
    """Quick plotting utility function."""
    plt.figure(figsize=(10, 6))
    plt.plot(data, **kwargs)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()