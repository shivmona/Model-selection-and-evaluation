import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Visualizer:
    """
    Handles data and model visualization.
    """
    
    def __init__(self):
        """Initialize the Visualizer class."""
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_cv_results(self, cv_results, metric):
        """
        Plot cross-validation results for a specific metric.
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results for each model
        metric : str
            Metric to visualize
            
        Returns:
        --------
        matplotlib.figure.Figure : The plot figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data for box plot
        data = []
        labels = []
        
        for model_name, model_results in cv_results.items():
            if metric in model_results:
                data.append(model_results[metric])
                labels.append(model_name)
        
        # Create box plot
        ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Add mean markers
        for i, d in enumerate(data):
            mean = np.mean(d)
            ax.scatter(i + 1, mean, marker='o', color='white', s=30, zorder=3)
            ax.text(i + 1.1, mean, f"{mean:.4f}", verticalalignment='center')
        
        # Set labels and title
        ax.set_title(f"Cross-Validation Results for {metric}")
        ax.set_ylabel(f"{metric} Value")
        ax.set_xlabel("Model")
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        return fig
    
    def plot_model_comparison_bar(self, results, metric):
        """
        Create a bar chart comparing models based on a metric.
        
        Parameters:
        -----------
        results : dict
            Cross-validation results for each model
        metric : str
            Metric to visualize
            
        Returns:
        --------
        matplotlib.figure.Figure : The plot figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data
        models = []
        mean_scores = []
        std_scores = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                models.append(model_name)
                mean_scores.append(np.mean(model_results[metric]))
                std_scores.append(np.std(model_results[metric]))
        
        # Sort by performance
        sorted_indices = np.argsort(mean_scores)
        if "error" in metric.lower():  # For error metrics, smaller is better
            sorted_indices = sorted_indices
        else:  # For score metrics, larger is better
            sorted_indices = sorted_indices[::-1]
            
        models = [models[i] for i in sorted_indices]
        mean_scores = [mean_scores[i] for i in sorted_indices]
        std_scores = [std_scores[i] for i in sorted_indices]
        
        # Create bar chart
        x = np.arange(len(models))
        bars = ax.bar(x, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(mean_scores):
            ax.text(i, v + std_scores[i] + 0.01, f"{v:.4f}", ha='center')
        
        # Set labels and title
        ax.set_title(f"Model Comparison - {metric}")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison_radar(self, results, metrics):
        """
        Create a radar chart comparing models across multiple metrics.
        
        Parameters:
        -----------
        results : dict
            Cross-validation results for each model
        metrics : list
            List of metrics to visualize
            
        Returns:
        --------
        matplotlib.figure.Figure : The plot figure
        """
        # Number of metrics (variables)
        n_metrics = len(metrics)
        
        # Check if there are enough metrics for a radar chart
        if n_metrics < 3:
            # Fall back to bar chart if not enough metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Radar chart requires at least 3 metrics", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Prepare the data
        model_names = list(results.keys())
        
        # Compute normalized scores for each model and metric
        normalized_scores = {}
        
        for metric in metrics:
            # Extract scores for this metric for all models
            metric_scores = {}
            for model in model_names:
                if metric in results[model]:
                    metric_scores[model] = np.mean(results[model][metric])
            
            # Determine if smaller is better (error metrics)
            smaller_is_better = "error" in metric.lower()
            
            # Find min and max 
            min_score = min(metric_scores.values())
            max_score = max(metric_scores.values())
            
            # Normalize scores to [0, 1]
            if min_score != max_score:  # Avoid division by zero
                for model in model_names:
                    if metric in results[model]:
                        score = np.mean(results[model][metric])
                        if smaller_is_better:
                            # Invert for error metrics
                            normalized_scores.setdefault(model, {})[metric] = 1 - (score - min_score) / (max_score - min_score)
                        else:
                            normalized_scores.setdefault(model, {})[metric] = (score - min_score) / (max_score - min_score)
            else:
                # If all models have the same score
                for model in model_names:
                    if metric in results[model]:
                        normalized_scores.setdefault(model, {})[metric] = 0.5
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Add labels for each metric
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Choose colors for each model
        colors = plt.cm.tab10.colors
        
        # Plot each model
        for i, model in enumerate(model_names):
            # Get values for this model
            values = [normalized_scores[model].get(metric, 0) for metric in metrics]
            values += values[:1]  # Close the polygon
            
            # Plot values
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # Set y-ticks
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Model Comparison Across Metrics", size=15)
        plt.tight_layout()
        
        return fig
    
    def plot_model_comparison_box(self, results, metric):
        """
        Create a box plot comparing cross-validation distributions across models.
        
        Parameters:
        -----------
        results : dict
            Cross-validation results for each model
        metric : str
            Metric to visualize
            
        Returns:
        --------
        matplotlib.figure.Figure : The plot figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for box plot
        data = []
        labels = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                data.append(model_results[metric])
                labels.append(model_name)
        
        # Sort models by performance
        if data:
            means = [np.mean(d) for d in data]
            
            # Determine sort order based on metric type
            if "error" in metric.lower():  # For error metrics, smaller is better
                sorted_indices = np.argsort(means)
            else:  # For score metrics, larger is better
                sorted_indices = np.argsort(means)[::-1]
                
            data = [data[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
        
        # Create box plot with seaborn for nicer styling
        sns.boxplot(data=data, orient='h', palette='viridis', ax=ax)
        
        # Set labels and title
        ax.set_title(f"Model Comparison - {metric} Distribution")
        ax.set_xlabel(metric)
        ax.set_yticklabels(labels)
        
        # Add mean markers
        for i, d in enumerate(data):
            mean = np.mean(d)
            ax.scatter(mean, i, marker='o', color='red', s=30, zorder=3)
            ax.text(mean, i + 0.1, f"{mean:.4f}", verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
