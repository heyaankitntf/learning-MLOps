import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Setting tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db") # Saves to a single database file

# 1. Generate some dummy demand data so you can run this locally
np.random.seed(42)
X = pd.DataFrame({
    'Price': np.random.randint(50, 200, 1000),
    'Discount': np.random.randint(0, 30, 1000),
    'Competitor_Pricing': np.random.randint(50, 200, 1000)
})
y = np.random.randint(10, 500, 1000) # Fake demand numbers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# THE MLFLOW PART STARTS HERE
# ==========================================

# Set the name of the whole project folder
mlflow.set_experiment("Demand_Forecasting_XGBoost")

# TUI/CODE: Start an MLflow "run". Think of this as clicking "Record" on a video camera.
with mlflow.start_run(run_name="n_estimators_150"):
    
    # Define some hyperparameters
    n_estimators = 150
    max_depth = 6
    
    # TUI/CODE 1: Log the hyperparameters to the text log
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train the model (normal code)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate (normal code)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae:.2f}")
    
    # TUI/CODE 2: Log the metric to the text log
    mlflow.log_metric("MAE", mae)
    
    # TUI/CODE 3: Log the actual model file automatically!
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Create a text summary
    summary = f"""
    Test Data Summary:
    - Mean Actual Demand: {y_test.mean():.2f}
    - Std Dev Actual Demand: {y_test.std():.2f}
    - Min Actual Demand: {y_test.min()}
    - Max Actual Demand: {y_test.max()}
    """
    # Log it as a .txt file
    mlflow.log_text(summary, "data_summary.txt")
    
    print("Model logged successfully!")

# ==========================================
# THE MLFLOW PART ENDS HERE
# ==========================================