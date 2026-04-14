import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error,
    confusion_matrix,
    f1_score
)

def calculate_rss(y_true, y_pred):
    """Calculate Residual Sum of Squares (RSS)"""
    return np.sum(np.square(y_true - y_pred))

def calculate_directional_metrics(y_true, y_pred):
    """
    Converts continuous price predictions into binary directional movements (Up=1, Down=0)
    to calculate classification metrics like Confusion Matrix and F-measure.
    """
    if len(y_true) < 2:
        return np.array([[0, 0], [0, 0]]), 0.0

    # Calculate direction of actual vs predicted (1 for price increase/flat, 0 for decrease)
    actual_direction = (np.diff(y_true) >= 0).astype(int)
    pred_direction = (np.diff(y_pred) >= 0).astype(int)
    
    conf_matrix = confusion_matrix(actual_direction, pred_direction, labels=[0, 1])
    f_measure = f1_score(actual_direction, pred_direction, average='weighted', zero_division=0)
    
    return conf_matrix, f_measure

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluates model performance returning RMSE, R2, RSS, MAPE, F-Measure, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rss = calculate_rss(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Classification Metrics (Directional)
    conf_matrix, f_measure = calculate_directional_metrics(y_true, y_pred)
    
    return {
        "RMSE": rmse,
        "R2_Score": r2,
        "RSS": rss,
        "MAPE": mape,
        "F_Measure": f_measure,
        "Confusion_Matrix": conf_matrix.tolist()
    }

def display_metrics(model_name, metrics):
    """
    Pretty-prints the metrics for a given model.
    """
    print(f"=== Performance Metrics for {model_name} ===")
    print(f"RMSE (Root Mean Squared Error):  {metrics['RMSE']:.4f}")
    print(f"R2 Error (R-squared Score):      {metrics['R2_Score']:.4f}")
    print(f"RSS (Residual Sum of Squares):   {metrics['RSS']:.4f}")
    print(f"MAPE:                            {metrics['MAPE']:.4%}")
    print(f"F-Measure (Trend Direction):     {metrics['F_Measure']:.4f}")
    print(f"Confusion Matrix (Trend Direction):\n{np.array(metrics['Confusion_Matrix'])}")
    print("=" * 45 + "\n")
