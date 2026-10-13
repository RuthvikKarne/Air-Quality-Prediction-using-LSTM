"""
LSTM Model Architecture and Pipeline for Air Quality Prediction
Supports univariate and multivariate time-series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os


def prepare_dataset(df, target_col='AQI', feature_cols=None, lookback=14, test_split=0.2):
    """
    Normalizes data and converts it into supervised sliding window sequences for LSTM.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with time series data.
        target_col (str): Column to predict.
        feature_cols (list): List of feature columns. If None, univariate [target_col].
        lookback (int): Number of prior time steps to use as input.
        test_split (float): Ratio of data to reserve for testing.
        
    Returns:
        dict: Processed sequences, scalers, and metadata.
    """
    if feature_cols is None or len(feature_cols) == 0:
        feature_cols = [target_col]
    elif target_col not in feature_cols:
        feature_cols = [target_col] + [c for c in feature_cols if c != target_col]
        
    # Ensure target_col is at index 0 for easy inverse transform
    sorted_features = [target_col] + [c for c in feature_cols if c != target_col]
    
    clean_df = df[sorted_features].dropna().copy()
    raw_values = clean_df.values
    
    # Feature scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(raw_values)
    
    # Target-specific scaler for inverse transformations
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(clean_df[[target_col]].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_values)):
        X.append(scaled_values[i - lookback:i, :])
        y.append(scaled_values[i, 0])  # target is at index 0
        
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(X)
    train_size = int(n_samples * (1 - test_split))
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'features': sorted_features,
        'target_col': target_col,
        'lookback': lookback,
        'clean_df': clean_df,
        'train_size': train_size
    }


def build_lstm_model(input_shape, units_l1=64, units_l2=32, dropout=0.2, learning_rate=0.001):
    """
    Builds and compiles a 2-layer stacked LSTM network for regression forecasting.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units_l1, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
        Dropout(dropout),
        LSTM(units_l2, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def calculate_metrics(y_true, y_pred, tolerance_pct=0.10):
    """
    Calculates comprehensive loss and accuracy metrics:
    - MSE, RMSE, MAE, MAPE
    - R² Score (Standard regression accuracy / explained variance)
    - Tolerance Accuracy (% within tolerance percentage of true value)
    - Directional Accuracy (Percentage of correct movement directions)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid zero division in MAPE
    denom = np.where(np.abs(y_true) < 1e-5, 1e-5, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    
    # R2 Score (Accuracy score in regression)
    r2 = r2_score(y_true, y_pred)
    
    # Percentage within tolerance (e.g. ±10%)
    rel_errors = np.abs((y_true - y_pred) / denom)
    tolerance_acc = np.mean(rel_errors <= tolerance_pct) * 100
    
    # Mean Directional Accuracy (Did model correctly predict up/down trend?)
    if len(y_true) > 1:
        dir_true = np.diff(y_true) > 0
        dir_pred = np.diff(y_pred) > 0
        directional_acc = np.mean(dir_true == dir_pred) * 100
    else:
        directional_acc = 100.0
        
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2_Score': float(r2),
        'R2_Accuracy_Pct': float(max(0.0, r2 * 100.0)),
        'Tolerance_Accuracy': float(tolerance_acc),
        'Directional_Accuracy': float(directional_acc)
    }


def forecast_future(model, last_sequence, target_scaler, feature_scaler, steps=14, n_features=1):
    """
    Autoregressively generates predictions for the next `steps` future days.
    """
    current_seq = np.copy(last_sequence)  # shape: (lookback, n_features)
    future_preds = []
    
    for _ in range(steps):
        # Reshape for model input: (1, lookback, n_features)
        pred_scaled = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0, 0]
        future_preds.append(pred_scaled)
        
        # Shift sequence window
        new_row = np.copy(current_seq[-1, :])
        new_row[0] = pred_scaled  # update target value (index 0)
        current_seq = np.vstack([current_seq[1:], new_row])
        
    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds_orig = target_scaler.inverse_transform(future_preds).flatten()
    return future_preds_orig


def get_aqi_category(aqi_val):
    """
    Returns the standard EPA/WHO AQI category, color code, and health advisory.
    """
    if aqi_val <= 50:
        return {"category": "Good", "color": "#00E400", "textColor": "#000", "description": "Air quality is satisfactory and poses little or no risk."}
    elif aqi_val <= 100:
        return {"category": "Moderate", "color": "#FFFF00", "textColor": "#000", "description": "Air quality is acceptable; however, sensitive individuals may experience minor symptoms."}
    elif aqi_val <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "color": "#FF7E00", "textColor": "#fff", "description": "Members of sensitive groups may experience health effects; general public is unlikely to be affected."}
    elif aqi_val <= 200:
        return {"category": "Unhealthy", "color": "#FF0000", "textColor": "#fff", "description": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."}
    elif aqi_val <= 300:
        return {"category": "Very Unhealthy", "color": "#8F3F97", "textColor": "#fff", "description": "Health alert: everyone may experience more serious health effects."}
    else:
        return {"category": "Hazardous", "color": "#7E0023", "textColor": "#fff", "description": "Health warning of emergency conditions: entire population is likely to be affected."}
