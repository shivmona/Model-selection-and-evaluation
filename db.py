import os
import pickle
import base64
import datetime
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define database models
class Dataset(Base):
    """Model for storing uploaded datasets"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    rows = Column(Integer)
    columns = Column(Integer)
    data_binary = Column(LargeBinary)  # Store pickled dataframe
    
    @property
    def data(self):
        """Return the unpickled dataframe"""
        if self.data_binary:
            return pickle.loads(self.data_binary)
        return None
    
    @data.setter
    def data(self, df):
        """Pickle the dataframe for storage"""
        if isinstance(df, pd.DataFrame):
            self.data_binary = pickle.dumps(df)
            self.rows = df.shape[0]
            self.columns = df.shape[1]
        else:
            raise ValueError("Data must be a pandas DataFrame")

class Model(Base):
    """Model for storing trained ML models"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # Classification or Regression
    algorithm = Column(String(50), nullable=False)  # e.g., Logistic Regression, Random Forest
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    parameters = Column(Text)  # JSON string of model parameters
    model_binary = Column(LargeBinary)  # Store pickled model
    dataset_id = Column(Integer)  # Reference to the dataset used
    
    @property
    def model(self):
        """Return the unpickled model"""
        if self.model_binary:
            return pickle.loads(self.model_binary)
        return None
    
    @model.setter
    def model(self, model_obj):
        """Pickle the model for storage"""
        self.model_binary = pickle.dumps(model_obj)
    
    @property
    def params(self):
        """Return the model parameters as a dictionary"""
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    @params.setter
    def params(self, params_dict):
        """Convert parameters dictionary to JSON string"""
        if isinstance(params_dict, dict):
            self.parameters = json.dumps(params_dict)
        else:
            raise ValueError("Parameters must be a dictionary")

class EvaluationResult(Base):
    """Model for storing model evaluation results"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    metric_name = Column(String(50), nullable=False)  # e.g., Accuracy, F1, R2
    metric_value = Column(Float, nullable=False)
    cv_fold = Column(Integer)  # Optional: For cross-validation results
    
    # For storing additional metadata about the evaluation
    evaluation_type = Column(String(50))  # e.g., Cross-validation, Test set, Validation set
    dataset_split = Column(String(20))  # e.g., train, test, validation

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Database session context manager
def get_db():
    """Get a database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Functions for database operations
def save_dataset(name, data, description=None):
    """
    Save a dataset to the database
    
    Parameters:
    -----------
    name : str
        Name of the dataset
    data : pandas.DataFrame
        The dataset to save
    description : str, optional
        Description of the dataset
        
    Returns:
    --------
    int : The ID of the saved dataset
    """
    db = SessionLocal()
    try:
        dataset = Dataset(
            name=name,
            description=description
        )
        dataset.data = data
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset.id
    finally:
        db.close()

def save_model(name, model_obj, model_type, algorithm, params, dataset_id=None):
    """
    Save a trained model to the database
    
    Parameters:
    -----------
    name : str
        Name of the model
    model_obj : object
        Trained model object
    model_type : str
        Type of model (classification or regression)
    algorithm : str
        Name of the algorithm
    params : dict
        Model parameters
    dataset_id : int, optional
        ID of the dataset used to train the model
        
    Returns:
    --------
    int : The ID of the saved model
    """
    db = SessionLocal()
    try:
        model = Model(
            name=name,
            model_type=model_type,
            algorithm=algorithm,
            dataset_id=dataset_id
        )
        model.model = model_obj
        model.params = params
        db.add(model)
        db.commit()
        db.refresh(model)
        return model.id
    finally:
        db.close()

def save_evaluation_result(model_id, metric_name, metric_value, 
                          evaluation_type="Test", dataset_split="test", cv_fold=None):
    """
    Save a model evaluation result to the database
    
    Parameters:
    -----------
    model_id : int
        ID of the model
    metric_name : str
        Name of the metric
    metric_value : float
        Value of the metric
    evaluation_type : str, optional
        Type of evaluation
    dataset_split : str, optional
        Which dataset split was used
    cv_fold : int, optional
        Cross-validation fold number
        
    Returns:
    --------
    int : The ID of the saved evaluation result
    """
    db = SessionLocal()
    try:
        result = EvaluationResult(
            model_id=model_id,
            metric_name=metric_name,
            metric_value=metric_value,
            evaluation_type=evaluation_type,
            dataset_split=dataset_split,
            cv_fold=cv_fold
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result.id
    finally:
        db.close()

def get_all_datasets():
    """
    Get all datasets from the database
    
    Returns:
    --------
    list : List of dataset records
    """
    db = SessionLocal()
    try:
        return db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    finally:
        db.close()

def get_dataset(dataset_id):
    """
    Get a dataset by ID
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset
        
    Returns:
    --------
    Dataset : Dataset record
    """
    db = SessionLocal()
    try:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    finally:
        db.close()

def get_all_models():
    """
    Get all models from the database
    
    Returns:
    --------
    list : List of model records
    """
    db = SessionLocal()
    try:
        return db.query(Model).order_by(Model.created_at.desc()).all()
    finally:
        db.close()

def get_model(model_id):
    """
    Get a model by ID
    
    Parameters:
    -----------
    model_id : int
        ID of the model
        
    Returns:
    --------
    Model : Model record
    """
    db = SessionLocal()
    try:
        return db.query(Model).filter(Model.id == model_id).first()
    finally:
        db.close()

def get_model_evaluation_results(model_id):
    """
    Get evaluation results for a model
    
    Parameters:
    -----------
    model_id : int
        ID of the model
        
    Returns:
    --------
    list : List of evaluation result records
    """
    db = SessionLocal()
    try:
        return db.query(EvaluationResult).filter(
            EvaluationResult.model_id == model_id
        ).order_by(EvaluationResult.created_at.desc()).all()
    finally:
        db.close()