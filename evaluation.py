import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                            r2_score, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit

class Evaluator:
    """
    Handles model evaluation and cross-validation.
    """
    
    def __init__(self):
        """Initialize the Evaluator class."""
        # Define available metrics
        self.classification_metrics = {
            "Accuracy": self._accuracy,
            "Precision": self._precision,
            "Recall": self._recall,
            "F1 Score": self._f1_score,
            "ROC AUC Score": self._roc_auc_score
        }
        
        self.regression_metrics = {
            "R² Score": self._r2_score,
            "Mean Absolute Error": self._mean_absolute_error,
            "Mean Squared Error": self._mean_squared_error,
            "Root Mean Squared Error": self._root_mean_squared_error
        }
    
    def _accuracy(self, y_true, y_pred):
        """Calculate accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def _precision(self, y_true, y_pred):
        """Calculate precision score."""
        return precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def _recall(self, y_true, y_pred):
        """Calculate recall score."""
        return recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def _f1_score(self, y_true, y_pred):
        """Calculate F1 score."""
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def _roc_auc_score(self, y_true, y_pred_proba):
        """Calculate ROC AUC score."""
        # Check if model has predict_proba method
        if hasattr(self.current_model, 'predict_proba'):
            try:
                # For binary classification
                if len(np.unique(y_true)) == 2:
                    return roc_auc_score(y_true, y_pred_proba)
                # For multi-class classification
                else:
                    return roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception:
                # If ROC AUC cannot be calculated (e.g. missing classes)
                return np.nan
        return np.nan
    
    def _r2_score(self, y_true, y_pred):
        """Calculate R² score."""
        return r2_score(y_true, y_pred)
    
    def _mean_absolute_error(self, y_true, y_pred):
        """Calculate mean absolute error."""
        return mean_absolute_error(y_true, y_pred)
    
    def _mean_squared_error(self, y_true, y_pred):
        """Calculate mean squared error."""
        return mean_squared_error(y_true, y_pred)
    
    def _root_mean_squared_error(self, y_true, y_pred):
        """Calculate root mean squared error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def get_cv_splitter(self, cv_strategy, n_splits=5):
        """
        Get a cross-validation splitter based on the strategy.
        
        Parameters:
        -----------
        cv_strategy : str
            Cross-validation strategy
        n_splits : int, default=5
            Number of splits (folds)
            
        Returns:
        --------
        object : Cross-validation splitter
        """
        if cv_strategy == "K-Fold":
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif cv_strategy == "Stratified K-Fold":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif cv_strategy == "Leave-One-Out":
            return LeaveOneOut()
        elif cv_strategy == "Time Series Split":
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    
    def cross_validate(self, model, X, y, cv_strategy="K-Fold", n_splits=5, 
                      metrics=None, problem_type="classification"):
        """
        Perform cross-validation and calculate metrics.
        
        Parameters:
        -----------
        model : object
            Model to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        cv_strategy : str, default="K-Fold"
            Cross-validation strategy
        n_splits : int, default=5
            Number of splits for cross-validation
        metrics : list, default=None
            List of metric names to compute
        problem_type : str, default="classification"
            Type of problem ('classification' or 'regression')
            
        Returns:
        --------
        dict : Cross-validation results for each metric
        """
        # Store the model
        self.current_model = model
        
        # Get metrics to compute
        if metrics is None:
            if problem_type == "classification":
                metrics = ["Accuracy", "F1 Score"]
            else:
                metrics = ["R² Score", "Mean Squared Error"]
        
        # Get metric functions
        if problem_type == "classification":
            metric_funcs = {m: self.classification_metrics[m] for m in metrics 
                          if m in self.classification_metrics}
        else:
            metric_funcs = {m: self.regression_metrics[m] for m in metrics 
                          if m in self.regression_metrics}
        
        # Get CV splitter
        cv = self.get_cv_splitter(cv_strategy, n_splits)
        
        # Initialize results dictionary
        results = {metric: [] for metric in metric_funcs.keys()}
        
        # Perform cross-validation
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            for metric_name, metric_func in metric_funcs.items():
                if metric_name == "ROC AUC Score" and hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    # For binary classification
                    if len(np.unique(y)) == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    score = metric_func(y_test, y_pred_proba)
                else:
                    score = metric_func(y_test, y_pred)
                
                results[metric_name].append(score)
        
        return results
