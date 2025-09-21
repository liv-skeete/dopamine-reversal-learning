"""
Statistical Analysis Module
==========================

Statistical analysis tools for dopamine research data.
Provides functions for hypothesis testing, effect size calculation,
and statistical summary of experimental results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import warnings


class StatisticalAnalyzer:
    """Statistical analysis tools for dopamine research experiments."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
    
    def compute_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """
        Compute descriptive statistics for a dataset.
        
        Args:
            data: List of numerical values
            
        Returns:
            Dictionary of descriptive statistics
        """
        if not data:
            return {}
            
        data_array = np.array(data)
        return {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'variance': np.var(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'range': np.max(data_array) - np.min(data_array),
            'q1': np.percentile(data_array, 25),
            'q3': np.percentile(data_array, 75),
            'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
            'skewness': stats.skew(data_array),
            'kurtosis': stats.kurtosis(data_array),
            'n': len(data_array)
        }
    
    def t_test(self, group1: List[float], group2: List[float], 
              alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            Dictionary with t-test results
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_ind(group1, group2, 
                                            alternative=alternative)
        
        effect_size = self._compute_cohens_d(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'alpha': self.alpha,
            'n1': len(group1),
            'n2': len(group2),
            'alternative': alternative
        }
    
    def paired_t_test(self, before: List[float], after: List[float],
                     alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform paired samples t-test.
        
        Args:
            before: Measurements before intervention
            after: Measurements after intervention
            alternative: Alternative hypothesis
            
        Returns:
            Dictionary with paired t-test results
        """
        if len(before) != len(after):
            raise ValueError("Before and after arrays must have the same length")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_rel(before, after, 
                                            alternative=alternative)
        
        effect_size = self._compute_cohens_d(before, after, paired=True)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'alpha': self.alpha,
            'n': len(before),
            'alternative': alternative
        }
    
    def anova(self, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform one-way ANOVA.
        
        Args:
            groups: Dictionary of group names to data lists
            
        Returns:
            Dictionary with ANOVA results
        """
        group_data = list(groups.values())
        group_names = list(groups.keys())
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_stat, p_value = stats.f_oneway(*group_data)
        
        # Post-hoc tests if significant
        post_hoc = {}
        if p_value < self.alpha and len(group_names) > 2:
            post_hoc = self._tukey_hsd(groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'group_means': {name: np.mean(data) for name, data in groups.items()},
            'group_sizes': {name: len(data) for name, data in groups.items()},
            'post_hoc': post_hoc
        }
    
    def correlation_analysis(self, x: List[float], y: List[float]) -> Dict[str, Any]:
        """
        Perform correlation analysis between two variables.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Dictionary with correlation results
        """
        if len(x) != len(y):
            raise ValueError("Variables must have the same length")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
        
        return {
            'pearson': {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < self.alpha
            },
            'spearman': {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < self.alpha
            },
            'n': len(x)
        }
    
    def _compute_cohens_d(self, group1: List[float], group2: List[float], 
                         paired: bool = False) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether groups are paired
            
        Returns:
            Cohen's d effect size
        """
        if paired:
            # For paired samples, use the standard deviation of the differences
            differences = np.array(group1) - np.array(group2)
            std_pooled = np.std(differences)
        else:
            # For independent samples, use pooled standard deviation
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            std_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        mean_diff = np.mean(group1) - np.mean(group2)
        return abs(mean_diff / std_pooled) if std_pooled != 0 else 0
    
    def _tukey_hsd(self, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform Tukey's HSD post-hoc test.
        
        Args:
            groups: Dictionary of group names to data lists
            
        Returns:
            Dictionary with Tukey HSD results
        """
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
        except ImportError:
            return {"error": "statsmodels not available for Tukey HSD"}
        
        # Prepare data for Tukey HSD
        all_data = []
        group_labels = []
        for group_name, data in groups.items():
            all_data.extend(data)
            group_labels.extend([group_name] * len(data))
        
        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=self.alpha)
        
        # Convert to readable format
        results = []
        for i in range(len(tukey_result._results_table.data)):
            if i == 0:  # Skip header
                continue
            row = tukey_result._results_table.data[i]
            results.append({
                'group1': row[0],
                'group2': row[1],
                'mean_diff': row[2],
                'p_value': row[3],
                'significant': row[4],
                'lower_ci': row[5],
                'upper_ci': row[6]
            })
        
        return {'tukey_hsd': results}
    
    def analyze_experiment(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of experiment results.
        
        Args:
            experiment_results: Dictionary containing experiment results
            
        Returns:
            Comprehensive statistical analysis
        """
        analysis = {}
        trials = experiment_results.get('trials', [])
        
        if not trials:
            return analysis
        
        df = pd.DataFrame(trials)
        
        # Analyze each numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis[col] = self.compute_descriptive_stats(df[col].tolist())
        
        # Group analysis by stimulus type if available
        if 'stimulus' in df.columns:
            stimulus_groups = {}
            for stimulus in df['stimulus'].unique():
                stimulus_data = df[df['stimulus'] == stimulus]
                stimulus_groups[stimulus] = {}
                
                for col in numeric_cols:
                    stimulus_groups[stimulus][col] = self.compute_descriptive_stats(
                        stimulus_data[col].tolist()
                    )
            
            analysis['by_stimulus'] = stimulus_groups
        
        return analysis


# Utility functions for quick statistical analysis
def quick_stats(data: List[float]) -> Dict[str, float]:
    """Quick descriptive statistics."""
    analyzer = StatisticalAnalyzer()
    return analyzer.compute_descriptive_stats(data)


def quick_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """Quick independent samples t-test."""
    analyzer = StatisticalAnalyzer()
    return analyzer.t_test(group1, group2)