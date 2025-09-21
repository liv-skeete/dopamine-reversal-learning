"""
Configuration Loader
===================

Utility functions for loading and managing configuration files
for dopamine research experiments.
"""

import json
import yaml
import os
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """Load and manage configuration files for experiments."""
    
    def __init__(self, config_dir: str = "./config"):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(config_dir, exist_ok=True)
    
    def load_config(self, config_name: str, config_type: str = "json") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration file (without extension)
            config_type: Type of configuration file ('json' or 'yaml')
            
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.{config_type}")
        
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if config_type == "json":
                    return json.load(f)
                elif config_type == "yaml":
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")
        except Exception as e:
            self.logger.error(f"Error loading config {config_path}: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any], config_name: str, 
                   config_type: str = "json") -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            config_name: Name of the configuration file (without extension)
            config_type: Type of configuration file ('json' or 'yaml')
            
        Returns:
            True if successful, False otherwise
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.{config_type}")
        
        try:
            with open(config_path, 'w') as f:
                if config_type == "json":
                    json.dump(config, f, indent=2)
                elif config_type == "yaml":
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")
            
            self.logger.info(f"Config saved to: {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config {config_path}: {e}")
            return False
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.
        
        Args:
            agent_type: Type of agent ('prediction_error', 'hedonic', 'incentive_salience')
            
        Returns:
            Agent configuration
        """
        return self.load_config(f"agent_{agent_type}")
    
    def get_experiment_config(self, experiment_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific experiment type.
        
        Args:
            experiment_type: Type of experiment
            
        Returns:
            Experiment configuration
        """
        return self.load_config(f"experiment_{experiment_type}")
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist."""
        default_configs = {
            'agent_prediction_error': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'description': 'Prediction Error Agent Configuration'
            },
            'agent_hedonic': {
                'baseline_hedonia': 0.0,
                'habituation_rate': 0.05,
                'sensitization_rate': 0.1,
                'description': 'Hedonic Agent Configuration'
            },
            'agent_incentive_salience': {
                'baseline_motivation': 0.1,
                'learning_rate': 0.2,
                'decay_rate': 0.05,
                'threshold': 0.5,
                'description': 'Incentive Salience Agent Configuration'
            },
            'experiment_default': {
                'num_trials': 100,
                'output_dir': './results',
                'save_results': True,
                'description': 'Default Experiment Configuration'
            }
        }
        
        for config_name, config_data in default_configs.items():
            config_path = os.path.join(self.config_dir, f"{config_name}.json")
            if not os.path.exists(config_path):
                self.save_config(config_data, config_name)
                self.logger.info(f"Created default config: {config_name}")


# Global configuration instance
_config_loader = None

def get_config_loader(config_dir: str = "./config") -> ConfigLoader:
    """Get or create a global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
        _config_loader.create_default_configs()
    return _config_loader

def load_config(config_name: str, config_type: str = "json") -> Dict[str, Any]:
    """Convenience function to load configuration."""
    loader = get_config_loader()
    return loader.load_config(config_name, config_type)

def save_config(config: Dict[str, Any], config_name: str, config_type: str = "json") -> bool:
    """Convenience function to save configuration."""
    loader = get_config_loader()
    return loader.save_config(config, config_name, config_type)