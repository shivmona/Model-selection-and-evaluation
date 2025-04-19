import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
import os
import datetime
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from model_manager import ModelManager
from evaluation import Evaluator
from visualization import Visualizer
import db

# Set page configuration
st.set_page_config(
    page_title="ML Model Selection & Evaluation Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = {}

# Title and intro
st.title("Machine Learning Model Selection & Evaluation Platform")
st.markdown("""
This platform helps you build, evaluate, and compare machine learning models.
Upload your data, preprocess it, select models, and analyze their performance.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Data Upload & Preprocessing", "Model Selection", "Cross-Validation & Evaluation", 
         "Model Comparison", "Hyperparameter Tuning", "Model Export", "Database Management"]
page = st.sidebar.radio("Go to", pages)

# Instantiate helper classes
data_processor = DataProcessor()
model_manager = ModelManager()
evaluator = Evaluator()
visualizer = Visualizer()

# 1. DATA UPLOAD & PREPROCESSING PAGE
if page == "Data Upload & Preprocessing":
    st.header("Data Upload & Preprocessing")
    
    # Data upload section
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success(f"Successfully uploaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Data info
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
            # Data preprocessing options
            st.subheader("Data Preprocessing")
            
            # Target column selection
            target_col = st.selectbox("Select target column", data.columns)
            
            # Feature columns selection
            feature_cols = st.multiselect("Select feature columns (leave empty to select all except target)", 
                                         [col for col in data.columns if col != target_col],
                                         default=[col for col in data.columns if col != target_col])
            
            if not feature_cols:  # If no features selected, use all except target
                feature_cols = [col for col in data.columns if col != target_col]
            
            # Preprocessing options
            st.markdown("#### Preprocessing Options")
            
            # Missing values handling
            missing_strategy = st.selectbox("Missing values strategy", 
                                          ["None", "Remove rows", "Mean/Mode imputation"])
            
            # Categorical encoding
            categorical_encoding = st.selectbox("Categorical encoding", 
                                              ["None", "One-Hot Encoding", "Label Encoding"])
            
            # Feature scaling
            scaling_method = st.selectbox("Feature scaling", 
                                        ["None", "StandardScaler", "MinMaxScaler"])
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            
            # Apply preprocessing
            if st.button("Apply Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    X, y, feature_names = data_processor.preprocess_data(
                        data, target_col, feature_cols, 
                        missing_strategy, categorical_encoding, scaling_method
                    )
                    
                    # Store in session state
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.feature_names = feature_names
                    st.session_state.target_col = target_col
                    st.session_state.preprocessed = True
                    
                    # Split data for later use
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Determine problem type
                    if len(np.unique(y)) <= 10:  # Simple heuristic for classification vs regression
                        st.session_state.problem_type = "classification"
                        st.info(f"Detected a classification problem with {len(np.unique(y))} classes")
                    else:
                        st.session_state.problem_type = "regression"
                        st.info("Detected a regression problem")
                    
                    st.success("Preprocessing completed!")
                    
                    # Show preprocessing summary
                    st.subheader("Preprocessing Summary")
                    st.write(f"Features: {len(feature_names)}")
                    st.write(f"Samples: {X.shape[0]}")
                    st.write(f"Training set: {X_train.shape[0]} samples")
                    st.write(f"Test set: {X_test.shape[0]} samples")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# 2. MODEL SELECTION PAGE
elif page == "Model Selection":
    st.header("Model Selection")
    
    if not st.session_state.preprocessed:
        st.warning("Please upload and preprocess data first!")
    else:
        st.subheader("Select Models to Train")
        
        # Show problem type
        problem_type = st.session_state.problem_type
        st.info(f"Current problem type: {problem_type}")
        
        # Get available models based on problem type
        available_models = model_manager.get_available_models(problem_type)
        
        # Create multiple tabs for different model categories
        tabs = st.tabs([model_type for model_type in available_models.keys()])
        
        selected_models = {}
        
        for i, (model_type, models) in enumerate(available_models.items()):
            with tabs[i]:
                st.markdown(f"### {model_type} Models")
                
                for model_name, model_class in models.items():
                    model_expander = st.expander(f"{model_name}")
                    with model_expander:
                        selected = st.checkbox(f"Include {model_name} in analysis", key=f"select_{model_name}")
                        
                        if selected:
                            # Get default parameters for this model
                            default_params = model_manager.get_default_params(model_name, problem_type)
                            custom_params = {}
                            
                            # Show parameters that can be configured
                            st.markdown("#### Configure Hyperparameters")
                            
                            for param, details in default_params.items():
                                param_type = details["type"]
                                default_value = details["default"]
                                desc = details.get("description", "")
                                
                                if param_type == "int":
                                    min_val = details.get("min", 1)
                                    max_val = details.get("max", 100)
                                    custom_params[param] = st.slider(
                                        f"{param} {desc}", min_val, max_val, default_value, key=f"{model_name}_{param}"
                                    )
                                elif param_type == "float":
                                    min_val = details.get("min", 0.0)
                                    max_val = details.get("max", 1.0)
                                    custom_params[param] = st.slider(
                                        f"{param} {desc}", min_val, max_val, default_value, 
                                        step=details.get("step", 0.01), key=f"{model_name}_{param}"
                                    )
                                elif param_type == "categorical":
                                    options = details.get("options", [default_value])
                                    custom_params[param] = st.selectbox(
                                        f"{param} {desc}", options, 
                                        index=options.index(default_value) if default_value in options else 0,
                                        key=f"{model_name}_{param}"
                                    )
                                elif param_type == "bool":
                                    custom_params[param] = st.checkbox(
                                        f"{param} {desc}", default_value, key=f"{model_name}_{param}"
                                    )
                            
                            selected_models[model_name] = custom_params
        
        if selected_models:
            if st.button("Train Selected Models"):
                with st.spinner("Training models... This may take a moment."):
                    # Clear previous models
                    st.session_state.models = {}
                    
                    # Train each selected model
                    for model_name, params in selected_models.items():
                        model = model_manager.create_model(model_name, params, problem_type)
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.models[model_name] = {
                            "model": model,
                            "params": params
                        }
                    
                    st.success(f"Successfully trained {len(selected_models)} models!")
                    
                    # Show a quick evaluation on test set
                    st.subheader("Quick Evaluation (Test Set)")
                    
                    test_results = {}
                    for model_name, model_info in st.session_state.models.items():
                        model = model_info["model"]
                        score = model.score(st.session_state.X_test, st.session_state.y_test)
                        test_results[model_name] = score
                    
                    # Convert results to DataFrame for nice display
                    result_df = pd.DataFrame({
                        "Model": test_results.keys(),
                        "Score": test_results.values()
                    })
                    result_df = result_df.sort_values(by="Score", ascending=False)
                    
                    # Determine what score means based on problem type
                    score_name = "Accuracy" if problem_type == "classification" else "R² Score"
                    result_df = result_df.rename(columns={"Score": score_name})
                    
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Basic visualization
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(result_df["Model"], result_df[score_name])
                    ax.set_title(f"Model Comparison ({score_name})")
                    ax.set_xlabel("Model")
                    ax.set_ylabel(score_name)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("Select at least one model to train")

# 3. CROSS-VALIDATION & EVALUATION PAGE
elif page == "Cross-Validation & Evaluation":
    st.header("Cross-Validation & Evaluation")
    
    if not st.session_state.preprocessed:
        st.warning("Please upload and preprocess data first!")
    elif not st.session_state.models:
        st.warning("Please train at least one model first!")
    else:
        st.subheader("Configure Cross-Validation")
        
        # Cross-validation settings
        cv_strategy = st.selectbox(
            "Cross-validation strategy", 
            ["K-Fold", "Stratified K-Fold", "Leave-One-Out", "Time Series Split"],
            0
        )
        
        # CV parameters
        if cv_strategy in ["K-Fold", "Stratified K-Fold", "Time Series Split"]:
            n_splits = st.slider("Number of splits", 2, 10, 5)
        else:
            n_splits = None
        
        # Metrics selection
        st.subheader("Select Performance Metrics")
        
        if st.session_state.problem_type == "classification":
            metrics = st.multiselect(
                "Evaluation metrics",
                ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
                ["Accuracy", "F1 Score"]
            )
        else:  # regression
            metrics = st.multiselect(
                "Evaluation metrics",
                ["R² Score", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error"],
                ["R² Score", "Mean Squared Error"]
            )
        
        # Model selection for evaluation
        st.subheader("Select Models to Evaluate")
        models_to_evaluate = st.multiselect(
            "Models",
            list(st.session_state.models.keys()),
            list(st.session_state.models.keys())
        )
        
        if models_to_evaluate and metrics and st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                cv_results = {}
                
                for model_name in models_to_evaluate:
                    model = st.session_state.models[model_name]["model"]
                    
                    # Get scores for each metric
                    scores = evaluator.cross_validate(
                        model, 
                        st.session_state.X, 
                        st.session_state.y,
                        cv_strategy=cv_strategy,
                        n_splits=n_splits,
                        metrics=metrics,
                        problem_type=st.session_state.problem_type
                    )
                    
                    cv_results[model_name] = scores
                
                # Store results in session state
                st.session_state.evaluation_results = cv_results
                
                # Display results in a table
                st.subheader("Cross-Validation Results")
                
                # Create a DataFrame for the results
                result_rows = []
                
                for model_name, model_scores in cv_results.items():
                    row = {"Model": model_name}
                    
                    for metric, values in model_scores.items():
                        row[f"{metric} (Mean)"] = np.mean(values)
                        row[f"{metric} (Std)"] = np.std(values)
                    
                    result_rows.append(row)
                
                result_df = pd.DataFrame(result_rows)
                st.dataframe(result_df, use_container_width=True)
                
                # Visualization of results
                st.subheader("Performance Visualization")
                
                # Create a plot for each metric
                tabs = st.tabs(metrics)
                
                for i, metric in enumerate(metrics):
                    with tabs[i]:
                        fig = visualizer.plot_cv_results(cv_results, metric)
                        st.pyplot(fig)
                
                st.success("Cross-validation completed!")

# 4. MODEL COMPARISON PAGE
elif page == "Model Comparison":
    st.header("Model Comparison")
    
    if not st.session_state.evaluation_results:
        st.warning("Please run cross-validation first!")
    else:
        st.subheader("Performance Comparison Dashboard")
        
        # Get the available metrics from the results
        results = st.session_state.evaluation_results
        first_model = list(results.keys())[0]
        available_metrics = list(results[first_model].keys())
        
        # Let user select metrics to compare
        selected_metrics = st.multiselect(
            "Select metrics to compare", 
            available_metrics,
            default=available_metrics[:2]  # Default to first two metrics
        )
        
        if selected_metrics:
            # Create visualization based on selected metrics
            tabs = st.tabs(["Bar Chart", "Radar Chart", "Box Plot"])
            
            with tabs[0]:
                st.subheader("Bar Chart Comparison")
                for metric in selected_metrics:
                    fig = visualizer.plot_model_comparison_bar(results, metric)
                    st.pyplot(fig)
            
            with tabs[1]:
                st.subheader("Radar Chart Comparison")
                fig = visualizer.plot_model_comparison_radar(results, selected_metrics)
                st.pyplot(fig)
            
            with tabs[2]:
                st.subheader("Box Plot Comparison")
                for metric in selected_metrics:
                    fig = visualizer.plot_model_comparison_box(results, metric)
                    st.pyplot(fig)
            
            # Show detailed comparison table
            st.subheader("Detailed Comparison Table")
            
            # Create a DataFrame with all metrics for all models
            comparison_data = []
            
            for model_name, model_metrics in results.items():
                row = {"Model": model_name}
                
                for metric, values in model_metrics.items():
                    row[f"{metric} (Mean)"] = np.mean(values)
                    row[f"{metric} (Std)"] = np.std(values)
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Find best model based on selected metrics
            st.subheader("Best Model Identification")
            
            best_models = {}
            for metric in selected_metrics:
                if st.session_state.problem_type == "classification":
                    # For classification metrics (higher is better)
                    best_model = max(results.items(), 
                                    key=lambda x: np.mean(x[1][metric]))[0]
                    best_value = np.mean(results[best_model][metric])
                    best_models[metric] = (best_model, best_value)
                else:
                    # For regression metrics like errors (lower is better)
                    if metric.lower() in ["mean absolute error", "mean squared error", "root mean squared error"]:
                        best_model = min(results.items(), 
                                        key=lambda x: np.mean(x[1][metric]))[0]
                        best_value = np.mean(results[best_model][metric])
                    else:  # For R² and similar (higher is better)
                        best_model = max(results.items(), 
                                        key=lambda x: np.mean(x[1][metric]))[0]
                        best_value = np.mean(results[best_model][metric])
                    best_models[metric] = (best_model, best_value)
            
            # Show best models
            for metric, (model, value) in best_models.items():
                st.info(f"Best model for {metric}: **{model}** with value {value:.4f}")
            
            # Store best models in session state
            st.session_state.best_models = best_models

# 5. HYPERPARAMETER TUNING PAGE
elif page == "Hyperparameter Tuning":
    st.header("Hyperparameter Tuning")
    
    if not st.session_state.preprocessed:
        st.warning("Please upload and preprocess data first!")
    elif not st.session_state.models:
        st.warning("Please train at least one model first!")
    else:
        st.subheader("Select Model to Tune")
        
        # Model selection for tuning
        model_to_tune = st.selectbox(
            "Model", 
            list(st.session_state.models.keys())
        )
        
        if model_to_tune:
            # Get tunable parameters for this model
            tunable_params = model_manager.get_tunable_params(
                model_to_tune, 
                st.session_state.problem_type
            )
            
            st.subheader("Configure Parameter Search")
            
            # Search strategy
            search_strategy = st.selectbox(
                "Search strategy",
                ["Grid Search", "Random Search"],
                0
            )
            
            # CV folds
            cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
            
            # Parameter ranges
            st.subheader("Parameter Ranges")
            
            param_grid = {}
            for param, param_info in tunable_params.items():
                param_type = param_info["type"]
                
                if param_type == "int":
                    min_val = st.number_input(f"Min value for {param}", 
                                            value=param_info.get("min", 1), 
                                            step=1, 
                                            key=f"min_{param}")
                    max_val = st.number_input(f"Max value for {param}", 
                                            value=param_info.get("max", 10), 
                                            step=1, 
                                            key=f"max_{param}")
                    step = st.number_input(f"Step for {param}", 
                                        value=param_info.get("step", 1), 
                                        step=1, 
                                        key=f"step_{param}")
                    
                    param_grid[param] = list(range(min_val, max_val + 1, step))
                
                elif param_type == "float":
                    min_val = st.number_input(f"Min value for {param}", 
                                            value=param_info.get("min", 0.01), 
                                            step=0.01, 
                                            format="%.2f",
                                            key=f"min_{param}")
                    max_val = st.number_input(f"Max value for {param}", 
                                            value=param_info.get("max", 1.0), 
                                            step=0.01, 
                                            format="%.2f",
                                            key=f"max_{param}")
                    step = st.number_input(f"Step for {param}", 
                                        value=param_info.get("step", 0.1), 
                                        step=0.01, 
                                        format="%.2f",
                                        key=f"step_{param}")
                    
                    # Create a list of values from min to max with step
                    param_grid[param] = [round(min_val + i * step, 4) 
                                        for i in range(int((max_val - min_val) / step) + 1)]
                
                elif param_type == "categorical":
                    options = param_info.get("options", [])
                    selected_options = st.multiselect(
                        f"Options for {param}",
                        options,
                        default=options,
                        key=f"options_{param}"
                    )
                    
                    if selected_options:
                        param_grid[param] = selected_options
                
                elif param_type == "bool":
                    use_both = st.checkbox(f"Try both True and False for {param}", 
                                        value=True, 
                                        key=f"both_{param}")
                    
                    if use_both:
                        param_grid[param] = [True, False]
                    else:
                        value = st.checkbox(f"Value for {param}", 
                                          value=True, 
                                          key=f"value_{param}")
                        param_grid[param] = [value]
            
            # Display param grid preview
            st.subheader("Parameter Grid Preview")
            st.write(param_grid)
            
            # Scoring metric
            if st.session_state.problem_type == "classification":
                scoring = st.selectbox(
                    "Scoring metric",
                    ["accuracy", "precision", "recall", "f1", "roc_auc"],
                    0
                )
            else:  # regression
                scoring = st.selectbox(
                    "Scoring metric",
                    ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
                    0
                )
            
            # Number of iterations for random search
            if search_strategy == "Random Search":
                n_iter = st.slider("Number of iterations", 10, 100, 20)
            else:
                n_iter = None
            
            if st.button("Run Hyperparameter Tuning"):
                if not param_grid:
                    st.error("Please set at least one parameter to tune!")
                else:
                    with st.spinner("Running hyperparameter tuning... This might take a while."):
                        # Get base model
                        base_model = st.session_state.models[model_to_tune]["model"]
                        
                        # Run tuning
                        best_model, best_params, results = model_manager.tune_hyperparameters(
                            base_model,
                            param_grid,
                            st.session_state.X,
                            st.session_state.y,
                            cv=cv_folds,
                            scoring=scoring,
                            search_strategy=search_strategy,
                            n_iter=n_iter
                        )
                        
                        # Store tuned model
                        st.session_state.models[f"{model_to_tune} (Tuned)"] = {
                            "model": best_model,
                            "params": best_params
                        }
                        
                        # Display results
                        st.success("Hyperparameter tuning completed!")
                        
                        st.subheader("Best Parameters")
                        st.write(best_params)
                        
                        st.subheader("Best Score")
                        st.write(f"{scoring}: {results.best_score_}")
                        
                        # Compare with original model
                        st.subheader("Comparison with Original Model")
                        
                        # Get scores on test set
                        original_model = st.session_state.models[model_to_tune]["model"]
                        original_score = original_model.score(
                            st.session_state.X_test, 
                            st.session_state.y_test
                        )
                        
                        tuned_score = best_model.score(
                            st.session_state.X_test, 
                            st.session_state.y_test
                        )
                        
                        # Create comparison DataFrame
                        comparison_df = pd.DataFrame({
                            "Model": [model_to_tune, f"{model_to_tune} (Tuned)"],
                            "Test Score": [original_score, tuned_score]
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Visualize comparison
                        fig, ax = plt.subplots()
                        ax.bar(comparison_df["Model"], comparison_df["Test Score"])
                        ax.set_title("Model Performance Comparison")
                        ax.set_ylabel("Test Score")
                        plt.tight_layout()
                        st.pyplot(fig)

# 6. MODEL EXPORT PAGE
elif page == "Model Export":
    st.header("Model Export")
    
    if not st.session_state.models:
        st.warning("Please train at least one model first!")
    else:
        st.subheader("Select Model to Export")
        
        # Model selection for export
        model_to_export = st.selectbox(
            "Model", 
            list(st.session_state.models.keys())
        )
        
        if model_to_export:
            st.write("This will save the model as a .joblib file that can be used for predictions later.")
            
            if st.button("Export Model"):
                with st.spinner("Exporting model..."):
                    # Get model to export
                    model = st.session_state.models[model_to_export]["model"]
                    
                    # Serialize model
                    model_bytes = io.BytesIO()
                    joblib.dump(model, model_bytes)
                    model_bytes.seek(0)
                    
                    # Create file name
                    file_name = f"{model_to_export.replace(' ', '_').lower()}.joblib"
                    
                    # Offer download
                    st.download_button(
                        label="Download Model",
                        data=model_bytes,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                    
                    st.success(f"Model ready for download as {file_name}")
                    
                    # Show usage example
                    st.subheader("How to Use the Exported Model")
                    
                    code = f"""
                    ```python
                    import joblib
                    
                    # Load the model
                    model = joblib.load('{file_name}')
                    
                    # Example of making predictions
                    # X_new should have the same features as your training data
                    # predictions = model.predict(X_new)
                    ```
                    """
                    
                    st.markdown(code)

# 7. DATABASE MANAGEMENT PAGE
elif page == "Database Management":
    st.header("Database Management")
    
    # Create tabs for different database operations
    db_tabs = st.tabs(["Save Data & Models", "Load Data & Models", "View Saved Items"])
    
    # 1. SAVE TAB
    with db_tabs[0]:
        st.subheader("Save Current Dataset and Models")
        
        # Save dataset
        st.markdown("### Save Dataset")
        
        if st.session_state.data is not None:
            dataset_name = st.text_input("Dataset name", 
                                       value=f"Dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            dataset_desc = st.text_area("Dataset description (optional)", 
                                      "Dataset uploaded and preprocessed with ML Platform")
            
            if st.button("Save Dataset to Database"):
                with st.spinner("Saving dataset..."):
                    try:
                        dataset_id = db.save_dataset(dataset_name, st.session_state.data, dataset_desc)
                        st.success(f"Dataset saved successfully! (ID: {dataset_id})")
                    except Exception as e:
                        st.error(f"Error saving dataset: {str(e)}")
        else:
            st.warning("No dataset available to save. Please upload data first.")
        
        # Save models
        st.markdown("### Save Trained Models")
        
        if st.session_state.models:
            st.write(f"You have {len(st.session_state.models)} trained models available to save.")
            
            # Select models to save
            models_to_save = st.multiselect(
                "Select models to save",
                list(st.session_state.models.keys()),
                list(st.session_state.models.keys())
            )
            
            model_save_name_prefix = st.text_input("Model name prefix", 
                                                value=f"Model_{datetime.datetime.now().strftime('%Y%m%d')}")
            
            if st.button("Save Selected Models to Database"):
                with st.spinner("Saving models..."):
                    saved_models = []
                    
                    for model_name in models_to_save:
                        try:
                            model_obj = st.session_state.models[model_name]["model"]
                            params = st.session_state.models[model_name]["params"]
                            
                            # Get dataset ID if available
                            dataset_id = None
                            
                            # Save model
                            model_id = db.save_model(
                                f"{model_save_name_prefix}_{model_name}",
                                model_obj,
                                st.session_state.problem_type,
                                model_name,
                                params,
                                dataset_id
                            )
                            
                            saved_models.append(model_name)
                            
                            # If we have evaluation results, save those too
                            if st.session_state.evaluation_results and model_name in st.session_state.evaluation_results:
                                for metric, values in st.session_state.evaluation_results[model_name].items():
                                    # Save each cross-validation fold result
                                    for i, value in enumerate(values):
                                        db.save_evaluation_result(
                                            model_id,
                                            metric,
                                            float(value),
                                            evaluation_type="Cross-Validation",
                                            dataset_split="train",
                                            cv_fold=i+1
                                        )
                                    
                                    # Also save mean value
                                    db.save_evaluation_result(
                                        model_id,
                                        f"{metric}_mean",
                                        float(np.mean(values)),
                                        evaluation_type="Cross-Validation",
                                        dataset_split="train"
                                    )
                        except Exception as e:
                            st.error(f"Error saving model {model_name}: {str(e)}")
                    
                    if saved_models:
                        st.success(f"Saved {len(saved_models)} models to database: {', '.join(saved_models)}")
                    else:
                        st.error("No models were saved successfully.")
        else:
            st.warning("No models available to save. Please train some models first.")
    
    # 2. LOAD TAB
    with db_tabs[1]:
        st.subheader("Load Saved Datasets and Models")
        
        # Load dataset
        st.markdown("### Load Dataset")
        
        # Get all saved datasets
        datasets = db.get_all_datasets()
        
        if datasets:
            dataset_options = [f"{ds.name} (ID: {ds.id}, Rows: {ds.rows}, Cols: {ds.columns})" for ds in datasets]
            selected_dataset = st.selectbox("Select dataset to load", dataset_options)
            
            if selected_dataset and st.button("Load Selected Dataset"):
                with st.spinner("Loading dataset..."):
                    try:
                        # Extract dataset ID from selection string
                        import re
                        dataset_id = int(re.search(r"ID: (\d+)", selected_dataset).group(1))
                        
                        # Load dataset
                        dataset = db.get_dataset(dataset_id)
                        if dataset:
                            st.session_state.data = dataset.data
                            st.session_state.preprocessed = False  # Reset preprocessing flag
                            st.success(f"Dataset '{dataset.name}' loaded successfully!")
                        else:
                            st.error("Dataset not found.")
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
        else:
            st.info("No saved datasets found in the database.")
        
        # Load models
        st.markdown("### Load Saved Models")
        
        # Get all saved models
        models = db.get_all_models()
        
        if models:
            model_options = [f"{model.name} (ID: {model.id}, Type: {model.algorithm})" for model in models]
            selected_model = st.selectbox("Select model to load", model_options)
            
            if selected_model and st.button("Load Selected Model"):
                with st.spinner("Loading model..."):
                    try:
                        # Extract model ID from selection string
                        import re
                        model_id = int(re.search(r"ID: (\d+)", selected_model).group(1))
                        
                        # Load model
                        model_record = db.get_model(model_id)
                        if model_record:
                            # Create entry in session state
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_obj = model_record.model
                            model_name = model_record.algorithm
                            params = model_record.params
                            
                            # Add the model to session state
                            st.session_state.models[model_name] = {
                                "model": model_obj,
                                "params": params
                            }
                            
                            st.success(f"Model '{model_record.name}' loaded successfully!")
                        else:
                            st.error("Model not found.")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No saved models found in the database.")
    
    # 3. VIEW TAB
    with db_tabs[2]:
        st.subheader("View Saved Items")
        
        # Tabs for different types of saved items
        view_tabs = st.tabs(["Datasets", "Models", "Evaluation Results"])
        
        # Datasets view
        with view_tabs[0]:
            st.markdown("### Saved Datasets")
            
            datasets = db.get_all_datasets()
            
            if datasets:
                # Create a DataFrame for nice display
                datasets_data = []
                for ds in datasets:
                    datasets_data.append({
                        "ID": ds.id,
                        "Name": ds.name,
                        "Rows": ds.rows,
                        "Columns": ds.columns,
                        "Created": ds.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Description": ds.description[:50] + "..." if ds.description and len(ds.description) > 50 else ds.description
                    })
                
                datasets_df = pd.DataFrame(datasets_data)
                st.dataframe(datasets_df, use_container_width=True)
            else:
                st.info("No datasets saved in the database.")
        
        # Models view
        with view_tabs[1]:
            st.markdown("### Saved Models")
            
            models = db.get_all_models()
            
            if models:
                # Create a DataFrame for nice display
                models_data = []
                for model in models:
                    models_data.append({
                        "ID": model.id,
                        "Name": model.name,
                        "Algorithm": model.algorithm,
                        "Type": model.model_type,
                        "Created": model.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Dataset ID": model.dataset_id
                    })
                
                models_df = pd.DataFrame(models_data)
                st.dataframe(models_df, use_container_width=True)
                
                # Model details expander
                with st.expander("View Model Details"):
                    model_id_for_details = st.number_input("Enter Model ID", min_value=1, value=1, step=1)
                    
                    if st.button("Show Details"):
                        model = db.get_model(model_id_for_details)
                        
                        if model:
                            st.write(f"### Model: {model.name}")
                            st.write(f"**Algorithm:** {model.algorithm}")
                            st.write(f"**Type:** {model.model_type}")
                            st.write(f"**Created:** {model.created_at}")
                            
                            st.write("#### Parameters:")
                            st.json(model.params)
                            
                            # Get evaluation results
                            eval_results = db.get_model_evaluation_results(model_id_for_details)
                            
                            if eval_results:
                                st.write("#### Evaluation Results:")
                                
                                eval_data = []
                                for result in eval_results:
                                    eval_data.append({
                                        "Metric": result.metric_name,
                                        "Value": result.metric_value,
                                        "Type": result.evaluation_type,
                                        "Split": result.dataset_split,
                                        "Fold": result.cv_fold
                                    })
                                
                                eval_df = pd.DataFrame(eval_data)
                                st.dataframe(eval_df)
                            else:
                                st.info("No evaluation results found for this model.")
                        else:
                            st.error("Model not found.")
            else:
                st.info("No models saved in the database.")
        
        # Evaluation results view
        with view_tabs[2]:
            st.markdown("### Evaluation Results")
            
            # Display model selection for viewing evaluation results
            models = db.get_all_models()
            
            if models:
                model_options = [f"{model.name} (ID: {model.id}, Type: {model.algorithm})" for model in models]
                selected_model = st.selectbox("Select model to view results", model_options, key="eval_results_model")
                
                if selected_model:
                    # Extract model ID from selection string
                    import re
                    model_id = int(re.search(r"ID: (\d+)", selected_model).group(1))
                    
                    # Get evaluation results
                    eval_results = db.get_model_evaluation_results(model_id)
                    
                    if eval_results:
                        # Create a DataFrame for nice display
                        eval_data = []
                        for result in eval_results:
                            eval_data.append({
                                "Metric": result.metric_name,
                                "Value": result.metric_value,
                                "Type": result.evaluation_type,
                                "Split": result.dataset_split,
                                "Fold": result.cv_fold,
                                "Created": result.created_at.strftime("%Y-%m-%d %H:%M:%S")
                            })
                        
                        eval_df = pd.DataFrame(eval_data)
                        st.dataframe(eval_df, use_container_width=True)
                        
                        # Visualization of results
                        st.subheader("Visualization")
                        
                        # Group by metrics for mean values
                        metrics_only = eval_df[eval_df["Metric"].str.contains("_mean") == False]
                        if not metrics_only.empty:
                            metrics = metrics_only["Metric"].unique()
                            
                            if len(metrics) > 0:
                                selected_metric = st.selectbox("Select metric to visualize", metrics)
                                
                                metric_data = metrics_only[metrics_only["Metric"] == selected_metric]
                                
                                if not metric_data.empty:
                                    # Create a bar chart of values across folds
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.bar(metric_data["Fold"].astype(str), metric_data["Value"])
                                    ax.set_title(f"{selected_metric} across CV Folds")
                                    ax.set_xlabel("Fold")
                                    ax.set_ylabel(selected_metric)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                    else:
                        st.info("No evaluation results found for this model.")
            else:
                st.info("No models saved in the database.")
