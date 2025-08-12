"""
Learning Stability Analysis and Quantification

This module provides rigorous metrics for quantifying learning stability,
convergence properties, and robustness of RL agents in the tutoring domain.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

@dataclass
class StabilityMetrics:
    """Comprehensive stability metrics for RL learning"""
    
    # Convergence metrics
    convergence_rate: float
    convergence_point: Optional[int]
    plateau_stability: float
    
    # Variance metrics
    reward_variance: float
    variance_trend: float
    coefficient_of_variation: float
    
    # Trend analysis
    learning_trend: float
    trend_consistency: float
    monotonicity_score: float
    
    # Robustness metrics
    outlier_frequency: float
    shock_recovery_time: float
    performance_volatility: float
    
    # Statistical measures
    autocorrelation: float
    stationarity_p_value: float
    regime_changes: int
    
    # Composite scores
    overall_stability_score: float
    learning_efficiency_score: float
    robustness_score: float

class LearningStabilityAnalyzer:
    """
    Comprehensive analyzer for learning stability and convergence
    
    Features:
    - Multiple stability metrics
    - Statistical significance testing
    - Regime change detection
    - Convergence analysis
    - Robustness quantification
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 convergence_threshold: float = 0.05,
                 outlier_threshold: float = 2.5,
                 min_convergence_length: int = 50):
        """
        Initialize stability analyzer
        
        Args:
            window_size: Window size for rolling statistics
            convergence_threshold: Threshold for convergence detection
            outlier_threshold: Standard deviations for outlier detection
            min_convergence_length: Minimum length for convergence plateau
        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.outlier_threshold = outlier_threshold
        self.min_convergence_length = min_convergence_length
    
    def analyze_learning_curve(self, 
                              rewards: List[float],
                              additional_metrics: Optional[Dict[str, List[float]]] = None) -> StabilityMetrics:
        """
        Comprehensive stability analysis of learning curve
        
        Args:
            rewards: List of episode rewards
            additional_metrics: Additional metrics to analyze (loss, exploration, etc.)
            
        Returns:
            Comprehensive stability metrics
        """
        rewards = np.array(rewards)
        
        if len(rewards) < self.window_size:
            warnings.warn(f"Insufficient data for analysis (need at least {self.window_size} points)")
            return self._empty_metrics()
        
        # Convergence analysis
        convergence_rate = self._calculate_convergence_rate(rewards)
        convergence_point = self._detect_convergence_point(rewards)
        plateau_stability = self._calculate_plateau_stability(rewards, convergence_point)
        
        # Variance analysis
        reward_variance = self._calculate_reward_variance(rewards)
        variance_trend = self._calculate_variance_trend(rewards)
        coefficient_of_variation = self._calculate_coefficient_of_variation(rewards)
        
        # Trend analysis
        learning_trend = self._calculate_learning_trend(rewards)
        trend_consistency = self._calculate_trend_consistency(rewards)
        monotonicity_score = self._calculate_monotonicity_score(rewards)
        
        # Robustness analysis
        outlier_frequency = self._calculate_outlier_frequency(rewards)
        shock_recovery_time = self._calculate_shock_recovery_time(rewards)
        performance_volatility = self._calculate_performance_volatility(rewards)
        
        # Statistical analysis
        autocorrelation = self._calculate_autocorrelation(rewards)
        stationarity_p_value = self._test_stationarity(rewards)
        regime_changes = self._detect_regime_changes(rewards)
        
        # Composite scores
        overall_stability_score = self._calculate_overall_stability(
            plateau_stability, reward_variance, trend_consistency
        )
        learning_efficiency_score = self._calculate_learning_efficiency(
            convergence_rate, learning_trend, monotonicity_score
        )
        robustness_score = self._calculate_robustness_score(
            outlier_frequency, shock_recovery_time, performance_volatility
        )
        
        return StabilityMetrics(
            convergence_rate=convergence_rate,
            convergence_point=convergence_point,
            plateau_stability=plateau_stability,
            reward_variance=reward_variance,
            variance_trend=variance_trend,
            coefficient_of_variation=coefficient_of_variation,
            learning_trend=learning_trend,
            trend_consistency=trend_consistency,
            monotonicity_score=monotonicity_score,
            outlier_frequency=outlier_frequency,
            shock_recovery_time=shock_recovery_time,
            performance_volatility=performance_volatility,
            autocorrelation=autocorrelation,
            stationarity_p_value=stationarity_p_value,
            regime_changes=regime_changes,
            overall_stability_score=overall_stability_score,
            learning_efficiency_score=learning_efficiency_score,
            robustness_score=robustness_score
        )
    
    def _calculate_convergence_rate(self, rewards: np.ndarray) -> float:
        """Calculate rate of convergence to stable performance"""
        if len(rewards) < 20:
            return 0.0
        
        # Smooth the curve
        smoothed = pd.Series(rewards).rolling(window=min(10, len(rewards)//4)).mean().dropna()
        
        # Calculate derivative (rate of change)
        diff = np.diff(smoothed)
        
        # Find where derivative becomes small (convergence)
        convergence_indices = np.where(np.abs(diff) < self.convergence_threshold)[0]
        
        if len(convergence_indices) == 0:
            return 0.0
        
        # Convergence rate as inverse of steps to convergence
        first_convergence = convergence_indices[0]
        return 1.0 / (first_convergence + 1) if first_convergence > 0 else 1.0
    
    def _detect_convergence_point(self, rewards: np.ndarray) -> Optional[int]:
        """Detect the point where learning converges"""
        if len(rewards) < self.min_convergence_length:
            return None
        
        smoothed = pd.Series(rewards).rolling(window=10).mean().dropna()
        
        # Look for sustained periods of low variance
        for i in range(len(smoothed) - self.min_convergence_length):
            segment = smoothed[i:i + self.min_convergence_length]
            if np.std(segment) < self.convergence_threshold:
                return i
        
        return None
    
    def _calculate_plateau_stability(self, rewards: np.ndarray, convergence_point: Optional[int]) -> float:
        """Calculate stability of the plateau after convergence"""
        if convergence_point is None or convergence_point >= len(rewards) - 10:
            return 0.0
        
        plateau = rewards[convergence_point:]
        
        if len(plateau) < 10:
            return 0.0
        
        # Stability as inverse of coefficient of variation
        mean_reward = np.mean(plateau)
        std_reward = np.std(plateau)
        
        if mean_reward == 0:
            return 0.0
        
        cv = std_reward / abs(mean_reward)
        return 1.0 / (1.0 + cv)
    
    def _calculate_reward_variance(self, rewards: np.ndarray) -> float:
        """Calculate normalized reward variance"""
        return float(np.var(rewards))
    
    def _calculate_variance_trend(self, rewards: np.ndarray) -> float:
        """Calculate trend in variance over time (decreasing variance indicates stabilization)"""
        if len(rewards) < self.window_size * 2:
            return 0.0
        
        # Calculate rolling variance
        rolling_var = pd.Series(rewards).rolling(window=self.window_size).var().dropna()
        
        if len(rolling_var) < 2:
            return 0.0
        
        # Linear trend in variance
        x = np.arange(len(rolling_var))
        slope, _, _, _, _ = stats.linregress(x, rolling_var)
        
        return float(slope)
    
    def _calculate_coefficient_of_variation(self, rewards: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        mean_reward = np.mean(rewards)
        if mean_reward == 0:
            return float('inf')
        
        return float(np.std(rewards) / abs(mean_reward))
    
    def _calculate_learning_trend(self, rewards: np.ndarray) -> float:
        """Calculate overall learning trend"""
        x = np.arange(len(rewards))
        slope, _, r_value, _, _ = stats.linregress(x, rewards)
        
        # Weight slope by R-squared for trend strength
        return float(slope * (r_value ** 2))
    
    def _calculate_trend_consistency(self, rewards: np.ndarray) -> float:
        """Calculate consistency of learning trend"""
        if len(rewards) < 20:
            return 0.0
        
        # Calculate trend in overlapping windows
        window_size = min(self.window_size, len(rewards) // 4)
        trends = []
        
        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i + window_size]
            x = np.arange(len(window))
            slope, _, _, _, _ = stats.linregress(x, window)
            trends.append(slope)
        
        # Consistency as inverse of variance in trends
        trend_variance = np.var(trends)
        return float(1.0 / (1.0 + trend_variance))
    
    def _calculate_monotonicity_score(self, rewards: np.ndarray) -> float:
        """Calculate how monotonic the learning is"""
        smoothed = pd.Series(rewards).rolling(window=10).mean().dropna()
        
        # Count monotonic increases vs decreases
        diff = np.diff(smoothed)
        increases = np.sum(diff > 0)
        decreases = np.sum(diff < 0)
        
        total_changes = increases + decreases
        if total_changes == 0:
            return 1.0
        
        # Monotonicity score favors consistent direction
        return float(abs(increases - decreases) / total_changes)
    
    def _calculate_outlier_frequency(self, rewards: np.ndarray) -> float:
        """Calculate frequency of outlier episodes"""
        z_scores = np.abs(stats.zscore(rewards))
        outliers = np.sum(z_scores > self.outlier_threshold)
        
        return float(outliers / len(rewards))
    
    def _calculate_shock_recovery_time(self, rewards: np.ndarray) -> float:
        """Calculate average time to recover from performance shocks"""
        z_scores = stats.zscore(rewards)
        outlier_indices = np.where(np.abs(z_scores) > self.outlier_threshold)[0]
        
        if len(outlier_indices) == 0:
            return 0.0
        
        recovery_times = []
        baseline = np.median(rewards)
        
        for idx in outlier_indices:
            if idx >= len(rewards) - 1:
                continue
            
            # Find recovery time (return to within 1 std of baseline)
            for recovery_idx in range(idx + 1, len(rewards)):
                if abs(rewards[recovery_idx] - baseline) < np.std(rewards):
                    recovery_times.append(recovery_idx - idx)
                    break
        
        return float(np.mean(recovery_times)) if recovery_times else float('inf')
    
    def _calculate_performance_volatility(self, rewards: np.ndarray) -> float:
        """Calculate performance volatility using rolling standard deviation"""
        rolling_std = pd.Series(rewards).rolling(window=min(20, len(rewards)//4)).std().dropna()
        
        return float(np.mean(rolling_std))
    
    def _calculate_autocorrelation(self, rewards: np.ndarray) -> float:
        """Calculate autocorrelation at lag 1"""
        if len(rewards) < 2:
            return 0.0
        
        correlation = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _test_stationarity(self, rewards: np.ndarray) -> float:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(rewards)
            return float(result[1])  # p-value
        except ImportError:
            # Fallback: simple trend test
            x = np.arange(len(rewards))
            _, _, _, p_value, _ = stats.linregress(x, rewards)
            return float(p_value)
    
    def _detect_regime_changes(self, rewards: np.ndarray) -> int:
        """Detect number of regime changes in learning"""
        if len(rewards) < 30:
            return 0
        
        # Use change point detection based on variance changes
        smoothed = pd.Series(rewards).rolling(window=10).mean().dropna()
        
        # Calculate rolling variance
        rolling_var = pd.Series(smoothed).rolling(window=15).var().dropna()
        
        # Find significant changes in variance
        var_changes = np.abs(np.diff(rolling_var))
        threshold = np.std(var_changes) * 2
        
        regime_changes = np.sum(var_changes > threshold)
        return int(regime_changes)
    
    def _calculate_overall_stability(self, 
                                   plateau_stability: float,
                                   reward_variance: float,
                                   trend_consistency: float) -> float:
        """Calculate overall stability composite score"""
        # Normalize variance component (inverse relationship)
        normalized_variance = 1.0 / (1.0 + reward_variance)
        
        # Weighted combination
        stability_score = (
            0.4 * plateau_stability +
            0.3 * normalized_variance +
            0.3 * trend_consistency
        )
        
        return float(np.clip(stability_score, 0.0, 1.0))
    
    def _calculate_learning_efficiency(self,
                                     convergence_rate: float,
                                     learning_trend: float,
                                     monotonicity_score: float) -> float:
        """Calculate learning efficiency composite score"""
        # Normalize components
        normalized_trend = np.tanh(learning_trend) if learning_trend > 0 else 0
        
        efficiency_score = (
            0.4 * convergence_rate +
            0.3 * normalized_trend +
            0.3 * monotonicity_score
        )
        
        return float(np.clip(efficiency_score, 0.0, 1.0))
    
    def _calculate_robustness_score(self,
                                  outlier_frequency: float,
                                  shock_recovery_time: float,
                                  performance_volatility: float) -> float:
        """Calculate robustness composite score"""
        # Normalize components (inverse relationships)
        normalized_outliers = 1.0 / (1.0 + outlier_frequency * 10)
        normalized_recovery = 1.0 / (1.0 + shock_recovery_time / 10)
        normalized_volatility = 1.0 / (1.0 + performance_volatility)
        
        robustness_score = (
            0.4 * normalized_outliers +
            0.3 * normalized_recovery +
            0.3 * normalized_volatility
        )
        
        return float(np.clip(robustness_score, 0.0, 1.0))
    
    def _empty_metrics(self) -> StabilityMetrics:
        """Return empty metrics for insufficient data"""
        return StabilityMetrics(
            convergence_rate=0.0,
            convergence_point=None,
            plateau_stability=0.0,
            reward_variance=0.0,
            variance_trend=0.0,
            coefficient_of_variation=0.0,
            learning_trend=0.0,
            trend_consistency=0.0,
            monotonicity_score=0.0,
            outlier_frequency=0.0,
            shock_recovery_time=0.0,
            performance_volatility=0.0,
            autocorrelation=0.0,
            stationarity_p_value=1.0,
            regime_changes=0,
            overall_stability_score=0.0,
            learning_efficiency_score=0.0,
            robustness_score=0.0
        )
    
    def compare_stability(self, 
                         metrics_list: List[StabilityMetrics],
                         labels: List[str]) -> Dict[str, Any]:
        """
        Compare stability across multiple learning runs
        
        Args:
            metrics_list: List of stability metrics
            labels: Labels for each metrics set
            
        Returns:
            Comparison analysis
        """
        if len(metrics_list) != len(labels):
            raise ValueError("Number of metrics and labels must match")
        
        comparison = {
            'labels': labels,
            'stability_ranking': [],
            'efficiency_ranking': [],
            'robustness_ranking': [],
            'statistical_tests': {}
        }
        
        # Rank by composite scores
        stability_scores = [(label, metrics.overall_stability_score) 
                          for label, metrics in zip(labels, metrics_list)]
        stability_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['stability_ranking'] = stability_scores
        
        efficiency_scores = [(label, metrics.learning_efficiency_score) 
                           for label, metrics in zip(labels, metrics_list)]
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['efficiency_ranking'] = efficiency_scores
        
        robustness_scores = [(label, metrics.robustness_score) 
                           for label, metrics in zip(labels, metrics_list)]
        robustness_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['robustness_ranking'] = robustness_scores
        
        # Statistical significance tests
        if len(metrics_list) >= 2:
            # Test differences in key metrics
            stability_values = [m.overall_stability_score for m in metrics_list]
            efficiency_values = [m.learning_efficiency_score for m in metrics_list]
            robustness_values = [m.robustness_score for m in metrics_list]
            
            # Pairwise comparisons (if more than 2 groups, use ANOVA)
            if len(metrics_list) == 2:
                _, p_stability = stats.ttest_ind([stability_values[0]], [stability_values[1]])
                _, p_efficiency = stats.ttest_ind([efficiency_values[0]], [efficiency_values[1]])
                _, p_robustness = stats.ttest_ind([robustness_values[0]], [robustness_values[1]])
            else:
                _, p_stability = stats.f_oneway(*[[s] for s in stability_values])
                _, p_efficiency = stats.f_oneway(*[[e] for e in efficiency_values])
                _, p_robustness = stats.f_oneway(*[[r] for r in robustness_values])
            
            comparison['statistical_tests'] = {
                'stability_p_value': float(p_stability),
                'efficiency_p_value': float(p_efficiency),
                'robustness_p_value': float(p_robustness)
            }
        
        return comparison
    
    def visualize_stability_analysis(self, 
                                   rewards: List[float],
                                   metrics: StabilityMetrics,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of stability analysis
        
        Args:
            rewards: Original reward data
            metrics: Computed stability metrics
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        rewards = np.array(rewards)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Learning Stability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning curve with convergence point
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.7, label='Raw rewards')
        smoothed = pd.Series(rewards).rolling(window=min(20, len(rewards)//10)).mean()
        ax.plot(smoothed, color='red', linewidth=2, label='Smoothed')
        
        if metrics.convergence_point:
            ax.axvline(metrics.convergence_point, color='green', linestyle='--', 
                      label=f'Convergence point: {metrics.convergence_point}')
        
        ax.set_title('Learning Curve')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Rolling variance
        ax = axes[0, 1]
        rolling_var = pd.Series(rewards).rolling(window=min(50, len(rewards)//5)).var()
        ax.plot(rolling_var)
        ax.set_title(f'Rolling Variance (trend: {metrics.variance_trend:.6f})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
        
        # 3. Stability metrics bar chart
        ax = axes[0, 2]
        metric_names = ['Overall\nStability', 'Learning\nEfficiency', 'Robustness']
        metric_values = [metrics.overall_stability_score, 
                        metrics.learning_efficiency_score,
                        metrics.robustness_score]
        bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'orange'])
        ax.set_title('Composite Stability Scores')
        ax.set_ylabel('Score (0-1)')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Autocorrelation and trend analysis
        ax = axes[1, 0]
        x = np.arange(len(rewards))
        slope, intercept, r_value, _, _ = stats.linregress(x, rewards)
        trend_line = slope * x + intercept
        
        ax.scatter(x[::10], rewards[::10], alpha=0.6, s=20)
        ax.plot(x, trend_line, 'r-', linewidth=2, 
               label=f'Trend (slope: {slope:.4f}, R²: {r_value**2:.3f})')
        ax.set_title('Trend Analysis')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Outlier analysis
        ax = axes[1, 1]
        z_scores = np.abs(stats.zscore(rewards))
        outlier_mask = z_scores > self.outlier_threshold
        
        ax.plot(rewards, 'b-', alpha=0.7, label='Normal episodes')
        ax.scatter(np.where(outlier_mask)[0], rewards[outlier_mask], 
                  color='red', s=50, label=f'Outliers ({np.sum(outlier_mask)})')
        ax.set_title(f'Outlier Detection (freq: {metrics.outlier_frequency:.3f})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Detailed metrics table
        ax = axes[1, 2]
        ax.axis('off')
        
        metric_text = f"""
        Convergence Rate: {metrics.convergence_rate:.4f}
        Plateau Stability: {metrics.plateau_stability:.4f}
        Coefficient of Variation: {metrics.coefficient_of_variation:.4f}
        Monotonicity Score: {metrics.monotonicity_score:.4f}
        Autocorrelation: {metrics.autocorrelation:.4f}
        Shock Recovery Time: {metrics.shock_recovery_time:.2f}
        Performance Volatility: {metrics.performance_volatility:.4f}
        Regime Changes: {metrics.regime_changes}
        Stationarity p-value: {metrics.stationarity_p_value:.4f}
        """
        
        ax.text(0.1, 0.9, metric_text, fontsize=10, fontfamily='monospace',
               verticalalignment='top', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Detailed Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
