import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelManager:
    """
    Manages the creation, configuration and tuning of machine learning models.
    """
    
    def __init__(self):
        """Initialize the ModelManager."""
        # Define available models for classification and regression
        self.classification_models = {
            "Linear Models": {
                "Logistic Regression": LogisticRegression
            },
            "Tree-Based Models": {
                "Decision Tree": DecisionTreeClassifier,
                "Random Forest": RandomForestClassifier,
                "Gradient Boosting": GradientBoostingClassifier
            },
            "Other Models": {
                "Support Vector Machine": SVC,
                "K-Nearest Neighbors": KNeighborsClassifier,
                "Gaussian Naive Bayes": GaussianNB
            }
        }
        
        self.regression_models = {
            "Linear Models": {
                "Linear Regression": LinearRegression,
                "Ridge Regression": Ridge,
                "Lasso Regression": Lasso,
                "ElasticNet": ElasticNet
            },
            "Tree-Based Models": {
                "Decision Tree": DecisionTreeRegressor,
                "Random Forest": RandomForestRegressor,
                "Gradient Boosting": GradientBoostingRegressor
            },
            "Other Models": {
                "Support Vector Machine": SVR,
                "K-Nearest Neighbors": KNeighborsRegressor
            }
        }
        
        # Default parameters for each model
        self.default_params = {
            # Classification models
            "Logistic Regression": {
                "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "description": "Regularization strength (smaller values = stronger regularization)"},
                "penalty": {"type": "categorical", "options": ["l2", "l1", "elasticnet", "none"], "default": "l2", "description": "Penalty type"},
                "max_iter": {"type": "int", "min": 100, "max": 1000, "default": 100, "description": "Maximum iterations"}
            },
            "Decision Tree": {
                "max_depth": {"type": "int", "min": 2, "max": 20, "default": 5, "description": "Maximum tree depth"},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "description": "Minimum samples required to split a node"},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "default": 1, "description": "Minimum samples required at a leaf node"}
            },
            "Random Forest": {
                "n_estimators": {"type": "int", "min": 10, "max": 200, "default": 100, "description": "Number of trees"},
                "max_depth": {"type": "int", "min": 2, "max": 20, "default": 5, "description": "Maximum tree depth"},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "description": "Minimum samples required to split a node"}
            },
            "Gradient Boosting": {
                "n_estimators": {"type": "int", "min": 10, "max": 200, "default": 100, "description": "Number of boosting stages"},
                "learning_rate": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.1, "description": "Learning rate"},
                "max_depth": {"type": "int", "min": 2, "max": 10, "default": 3, "description": "Maximum tree depth"}
            },
            "Support Vector Machine": {
                "C": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "description": "Regularization parameter"},
                "kernel": {"type": "categorical", "options": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf", "description": "Kernel type"},
                "gamma": {"type": "categorical", "options": ["scale", "auto"], "default": "scale", "description": "Kernel coefficient"}
            },
            "K-Nearest Neighbors": {
                "n_neighbors": {"type": "int", "min": 1, "max": 20, "default": 5, "description": "Number of neighbors"},
                "weights": {"type": "categorical", "options": ["uniform", "distance"], "default": "uniform", "description": "Weight function"},
                "p": {"type": "int", "min": 1, "max": 2, "default": 2, "description": "Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)"}
            },
            "Gaussian Naive Bayes": {
                "var_smoothing": {"type": "float", "min": 1e-10, "max": 1e-8, "default": 1e-9, "description": "Portion of the largest variance to add to variances"}
            },
            # Regression models
            "Linear Regression": {
                "fit_intercept": {"type": "bool", "default": True, "description": "Whether to fit the intercept"}
            },
            "Ridge Regression": {
                "alpha": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "description": "Regularization strength"},
                "solver": {"type": "categorical", "options": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "default": "auto", "description": "Solver to use"}
            },
            "Lasso Regression": {
                "alpha": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "description": "Regularization strength"},
                "max_iter": {"type": "int", "min": 100, "max": 2000, "default": 1000, "description": "Maximum number of iterations"}
            },
            "ElasticNet": {
                "alpha": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "description": "Regularization strength"},
                "l1_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "The ElasticNet mixing parameter (0=Ridge, 1=Lasso)"}
            }
        }
        
        # Define parameters to tune for each model during hyperparameter optimization
        self.tunable_params = {
            # Classification models
            "Logistic Regression": {
                "C": {"type": "float", "min": 0.001, "max": 100.0, "step": 0.1, "description": "Regularization strength"},
                "penalty": {"type": "categorical", "options": ["l2", "l1", "elasticnet", "none"], "description": "Penalty type"}
            },
            "Decision Tree": {
                "max_depth": {"type": "int", "min": 1, "max": 30, "step": 1, "description": "Maximum tree depth"},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "step": 1, "description": "Minimum samples to split"},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "step": 1, "description": "Minimum samples at leaf"}
            },
            "Random Forest": {
                "n_estimators": {"type": "int", "min": 10, "max": 300, "step": 10, "description": "Number of trees"},
                "max_depth": {"type": "int", "min": 2, "max": 20, "step": 1, "description": "Maximum tree depth"},
                "min_samples_split": {"type": "int", "min": 2, "max": 10, "step": 1, "description": "Minimum samples to split"}
            },
            "Gradient Boosting": {
                "n_estimators": {"type": "int", "min": 50, "max": 500, "step": 50, "description": "Number of boosting stages"},
                "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "step": 0.01, "description": "Learning rate"},
                "max_depth": {"type": "int", "min": 1, "max": 10, "step": 1, "description": "Maximum tree depth"}
            },
            "Support Vector Machine": {
                "C": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "description": "Regularization parameter"},
                "kernel": {"type": "categorical", "options": ["linear", "poly", "rbf", "sigmoid"], "description": "Kernel type"},
                "gamma": {"type": "categorical", "options": ["scale", "auto"], "description": "Kernel coefficient"}
            },
            "K-Nearest Neighbors": {
                "n_neighbors": {"type": "int", "min": 1, "max": 30, "step": 1, "description": "Number of neighbors"},
                "weights": {"type": "categorical", "options": ["uniform", "distance"], "description": "Weight function"},
                "p": {"type": "categorical", "options": [1, 2], "description": "Power parameter (1=Manhattan, 2=Euclidean)"}
            },
            "Gaussian Naive Bayes": {
                "var_smoothing": {"type": "float", "min": 1e-12, "max": 1e-6, "step": 1e-12, "description": "Variance smoothing"}
            },
            # Regression models
            "Linear Regression": {
                "fit_intercept": {"type": "bool", "description": "Whether to fit the intercept"}
            },
            "Ridge Regression": {
                "alpha": {"type": "float", "min": 0.01, "max": 100.0, "step": 0.1, "description": "Regularization strength"},
                "solver": {"type": "categorical", "options": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "description": "Solver to use"}
            },
            "Lasso Regression": {
                "alpha": {"type": "float", "min": 0.001, "max": 10.0, "step": 0.001, "description": "Regularization strength"},
                "max_iter": {"type": "int", "min": 100, "max": 5000, "step": 100, "description": "Maximum iterations"}
            },
            "ElasticNet": {
                "alpha": {"type": "float", "min": 0.001, "max": 10.0, "step": 0.001, "description": "Regularization strength"},
                "l1_ratio": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "description": "Mixing parameter"}
            }
        }
    
    def get_available_models(self, problem_type):
        """
        Get available models for a given problem type.
        
        Parameters:
        -----------
        problem_type : str
            Either 'classification' or 'regression'
            
        Returns:
        --------
        dict : Available models grouped by category
        """
        if problem_type == 'classification':
            return self.classification_models
        elif problem_type == 'regression':
            return self.regression_models
        else:
            raise ValueError("Problem type must be either 'classification' or 'regression'")
    
    def get_default_params(self, model_name, problem_type):
        """
        Get default parameters for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        problem_type : str
            Type of problem ('classification' or 'regression')
            
        Returns:
        --------
        dict : Default parameters for the model
        """
        if model_name in self.default_params:
            return self.default_params[model_name]
        else:
            return {}
    
    def get_tunable_params(self, model_name, problem_type):
        """
        Get parameters that can be tuned for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        problem_type : str
            Type of problem ('classification' or 'regression')
            
        Returns:
        --------
        dict : Tunable parameters for the model
        """
        if model_name in self.tunable_params:
            return self.tunable_params[model_name]
        else:
            return {}
    
    def create_model(self, model_name, params, problem_type):
        """
        Create a model instance with specified parameters.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to create
        params : dict
            Parameters for the model
        problem_type : str
            Type of problem ('classification' or 'regression')
            
        Returns:
        --------
        object : Instantiated model
        """
        # Find the model class
        model_class = None
        
        if problem_type == 'classification':
            for category, models in self.classification_models.items():
                if model_name in models:
                    model_class = models[model_name]
                    break
        else:  # regression
            for category, models in self.regression_models.items():
                if model_name in models:
                    model_class = models[model_name]
                    break
        
        if model_class is None:
            raise ValueError(f"Model '{model_name}' not found for problem type '{problem_type}'")
        
        # Create and return model instance
        try:
            return model_class(**params)
        except TypeError as e:
            # If there's an error with parameters, filter out any that aren't accepted
            import inspect
            valid_params = {}
            signature = inspect.signature(model_class.__init__)
            
            for param_name, param_value in params.items():
                if param_name in signature.parameters:
                    valid_params[param_name] = param_value
            
            return model_class(**valid_params)
    
    def tune_hyperparameters(self, base_model, param_grid, X, y, cv=5, scoring=None, 
                           search_strategy='Grid Search', n_iter=None):
        """
        Perform hyperparameter tuning.
        
        Parameters:
        -----------
        base_model : object
            Base model to tune
        param_grid : dict
            Parameter grid to search
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default=None
            Scoring metric
        search_strategy : str, default='Grid Search'
            Search strategy to use ('Grid Search' or 'Random Search')
        n_iter : int, default=None
            Number of iterations for random search
            
        Returns:
        --------
        tuple : (best_model, best_params, search_results)
        """
        if search_strategy == 'Grid Search':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
        else:  # Random Search
            if n_iter is None:
                n_iter = 10
                
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
        
        # Run the search
        search.fit(X, y)
        
        # Get the best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        return best_model, best_params, search
