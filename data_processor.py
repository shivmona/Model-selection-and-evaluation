import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    """
    Handles data preprocessing operations including:
    - Loading data
    - Handling missing values
    - Encoding categorical variables
    - Feature scaling
    """
    
    def __init__(self):
        """Initialize the DataProcessor class."""
        self.transformers = None
        self.feature_names = None
    
    def preprocess_data(self, data, target_col, feature_cols=None, 
                       missing_strategy="None", categorical_encoding="None", 
                       scaling_method="None"):
        """
        Preprocess the input data with specified methods.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The input data to preprocess
        target_col : str
            The name of the target column
        feature_cols : list, optional
            List of feature column names to use. If None, all columns except target are used.
        missing_strategy : str, default="None"
            Strategy for handling missing values. Options: "None", "Remove rows", "Mean/Mode imputation"
        categorical_encoding : str, default="None"
            Strategy for encoding categorical variables. Options: "None", "One-Hot Encoding", "Label Encoding"
        scaling_method : str, default="None"
            Method for feature scaling. Options: "None", "StandardScaler", "MinMaxScaler"
            
        Returns:
        --------
        X : numpy.ndarray
            Preprocessed feature matrix
        y : numpy.ndarray
            Target variable
        feature_names : list
            Names of features after preprocessing
        """
        # Copy data to avoid modifying the original
        data_copy = data.copy()
        
        # Extract target variable
        y = data_copy[target_col].values
        
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in data_copy.columns if col != target_col]
        
        X_df = data_copy[feature_cols]
        
        # Handle missing values
        if missing_strategy == "Remove rows":
            # Drop rows with missing values
            X_df = X_df.dropna()
            # Also remove corresponding target values
            y = y[X_df.index]
        
        # Identify numeric and categorical columns
        numeric_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Create preprocessing pipelines for each column type
        preprocessors = []
        
        # Numeric columns pipeline
        if numeric_cols:
            numeric_pipeline = []
            
            # Missing value imputation for numeric columns
            if missing_strategy == "Mean/Mode imputation":
                numeric_pipeline.append(('imputer', SimpleImputer(strategy='mean')))
            
            # Scaling for numeric columns
            if scaling_method == "StandardScaler":
                numeric_pipeline.append(('scaler', StandardScaler()))
            elif scaling_method == "MinMaxScaler":
                numeric_pipeline.append(('scaler', MinMaxScaler()))
            
            if numeric_pipeline:
                preprocessors.append(
                    ('numeric', Pipeline(numeric_pipeline), numeric_cols)
                )
        
        # Categorical columns pipeline
        if categorical_cols:
            categorical_pipeline = []
            
            # Missing value imputation for categorical columns
            if missing_strategy == "Mean/Mode imputation":
                categorical_pipeline.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            # Encoding for categorical columns
            if categorical_encoding == "One-Hot Encoding":
                categorical_pipeline.append(
                    ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
                )
            elif categorical_encoding == "Label Encoding":
                # LabelEncoder doesn't work in a Pipeline as it doesn't implement transform
                # Instead, we'll use OrdinalEncoder which is designed for this purpose
                categorical_pipeline.append(
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                )
            
            if categorical_pipeline:
                preprocessors.append(
                    ('categorical', Pipeline(categorical_pipeline), categorical_cols)
                )
        
        # Combine preprocessors if needed
        if preprocessors:
            self.transformers = ColumnTransformer(
                transformers=preprocessors,
                remainder='passthrough'
            )
            
            # Fit and transform the data
            X = self.transformers.fit_transform(X_df)
            
            # Get feature names after transformation
            self.feature_names = self._get_feature_names(
                self.transformers, X_df.columns.tolist(), 
                categorical_encoding if categorical_cols else "None"
            )
        else:
            # If no preprocessing required, convert directly to numpy array
            X = X_df.values
            self.feature_names = feature_cols
        
        return X, y, self.feature_names
    
    def _get_feature_names(self, column_transformer, original_features, categorical_encoding):
        """
        Get feature names after transformation from ColumnTransformer.
        
        Parameters:
        -----------
        column_transformer : ColumnTransformer
            The fitted column transformer
        original_features : list
            Original feature names
        categorical_encoding : str
            The encoding method used for categorical variables
            
        Returns:
        --------
        list : Feature names after transformation
        """
        result_features = []
        
        # Process each transformer
        for transformer_name, transformer, column_names in column_transformer.transformers_:
            if transformer_name == 'remainder':
                # Add any columns that were not transformed
                if transformer == 'passthrough':
                    remaining_cols = [col for col in original_features 
                                     if col not in column_transformer.get_feature_names_out()]
                    result_features.extend(remaining_cols)
                continue
                
            if transformer_name == 'numeric':
                # Numeric columns retain their original names
                result_features.extend(column_names)
            
            elif transformer_name == 'categorical':
                if categorical_encoding == "One-Hot Encoding":
                    # For one-hot encoding, generate new column names for each category
                    for col in column_names:
                        # Get all categories for this column
                        categories = transformer.named_steps['encoder'].categories_[column_names.index(col)]
                        for cat in categories:
                            result_features.append(f"{col}_{cat}")
                elif categorical_encoding == "Label Encoding":
                    # OrdinalEncoder keeps original column names
                    result_features.extend(column_names)
                else:
                    # Other methods keep original column names
                    result_features.extend(column_names)
        
        return result_features
    
    def transform_new_data(self, data):
        """
        Apply the same preprocessing to new data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New data to preprocess
            
        Returns:
        --------
        numpy.ndarray : Preprocessed data
        """
        if self.transformers is None:
            raise ValueError("The preprocessor has not been fitted yet.")
        
        return self.transformers.transform(data)
