"""
Standalone Script: 6. LSTM - Air Quality Prediction
Trains an LSTM model on Air Quality data and visualizes Loss and Accuracy curves.
Saves evaluation metrics, plots, and trained weights.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lstm_model import prepare_dataset, build_lstm_model, calculate_metrics, forecast_future

# Create output directories for plots and saved models
os.makedirs('static', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

def main():
    print("=" * 70)
    print("  6. LSTM - AIR QUALITY PREDICTION PIPELINE (TRAINING & VISUALIZATION)")
    print("=" * 70)
    
    # 1. Load Dataset
    data_path = 'air_quality_data.csv'
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Checking data.csv...")
        data_path = 'data.csv'
        target_col = 'Close'
        feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    else:
        target_col = 'AQI'
        feature_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'Temperature', 'Humidity']
        
    print(f"[*] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[*] Total rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"[*] Target variable: '{target_col}'")
    print(f"[*] Features included: {feature_cols}")
    
    # 2. Sequence Preprocessing
    lookback = 14
    test_split = 0.20
    print(f"[*] Preparing sequences with Lookback={lookback} days, Test Split={int(test_split*100)}%...")
    prep = prepare_dataset(df, target_col=target_col, feature_cols=feature_cols, lookback=lookback, test_split=test_split)
    
    X_train, y_train = prep['X_train'], prep['y_train']
    X_test, y_test = prep['X_test'], prep['y_test']
    
    print(f"[*] Training samples: {X_train.shape[0]}, Features per step: {X_train.shape[2]}")
    print(f"[*] Testing samples:  {X_test.shape[0]}")
    
    # 3. Build LSTM Model
    print("[*] Building Stacked LSTM Architecture...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units_l1=64,
        units_l2=32,
        dropout=0.2,
        learning_rate=0.001
    )
    model.summary()
    
    # 4. Train Model
    epochs = 25
    batch_size = 32
    print(f"[*] Training LSTM model for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 5. Evaluate on Test Set
    preds_scaled = model.predict(X_test)
    preds = prep['target_scaler'].inverse_transform(preds_scaled).flatten()
    actuals = prep['target_scaler'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    metrics = calculate_metrics(actuals, preds, tolerance_pct=0.10)
    
    print("\n" + "=" * 50)
    print("           MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Final Train Loss (MSE)      : {history.history['loss'][-1]:.6f}")
    print(f"  Final Val Loss (MSE)        : {history.history['val_loss'][-1]:.6f}")
    print(f"  Root Mean Sq Error (RMSE)   : {metrics['RMSE']:.4f}")
    print(f"  Mean Absolute Error (MAE)   : {metrics['MAE']:.4f}")
    print(f"  Mean Abs Percentage (MAPE)  : {metrics['MAPE']:.2f}%")
    print(f"  R^2 Score (Accuracy Score)  : {metrics['R2_Score']:.4f} ({metrics['R2_Accuracy_Pct']:.2f}%)")
    print(f"  Accuracy within +/-10% band : {metrics['Tolerance_Accuracy']:.2f}%")
    print(f"  Mean Directional Accuracy   : {metrics['Directional_Accuracy']:.2f}%")
    print("=" * 50)
    
    # 6. Generate and Save Visualizations ("in code visualization")
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Visualization 1: Loss Curve (Train vs Validation Loss)
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='#3b82f6', linewidth=2.2)
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#ef4444', linewidth=2.2, linestyle='--')
    plt.title('LSTM Training & Validation Loss Over Epochs', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = 'static/loss_curve.png'
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Loss Curve to: {loss_path}")
    
    # Visualization 2: Accuracy & Error Metric Curve (MAE across Epochs)
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['mae'], label='Training MAE', color='#10b981', linewidth=2.2)
    plt.plot(history.history['val_mae'], label='Validation MAE', color='#f59e0b', linewidth=2.2, linestyle='--')
    plt.title('LSTM Training & Validation MAE (Accuracy Tracking)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Absolute Error (Normalized)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = 'static/accuracy_curve.png'
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Accuracy/MAE Curve to: {acc_path}")
    
    # Visualization 3: Actual vs Predicted Time-Series
    plt.figure(figsize=(11, 5))
    test_indices = np.arange(len(actuals))
    plt.plot(test_indices, actuals, label='Actual Ground Truth', color='#0ea5e9', linewidth=2, marker='o', markersize=3, alpha=0.9)
    plt.plot(test_indices, preds, label='LSTM Predicted', color='#f43f5e', linewidth=2, linestyle='--', marker='x', markersize=3)
    plt.fill_between(test_indices, actuals, preds, color='#fda4af', alpha=0.2, label='Prediction Error Region')
    plt.title(f'LSTM Air Quality Forecast vs Actual Test Set (R^2 = {metrics["R2_Score"]:.3f})', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Test Sample Index (Days)', fontsize=12)
    plt.ylabel(f'{target_col} Level', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pred_path = 'static/actual_vs_predicted.png'
    plt.savefig(pred_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Prediction vs Actual to: {pred_path}")
    
    # Visualization 4: Residuals and Error Distribution
    residuals = actuals - preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(actuals, preds, alpha=0.6, color='#6366f1', edgecolors='none')
    min_val, max_val = min(min(actuals), min(preds)), max(max(actuals), max(preds))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit (y = x)')
    ax1.set_title('Actual vs Predicted Scatter Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Actual Value')
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(residuals, bins=25, color='#8b5cf6', edgecolor='black', alpha=0.7, density=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    ax2.set_title(f'Residuals Distribution (Mean={np.mean(residuals):.2f})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Prediction Error (Actual - Predicted)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    res_path = 'static/residuals_distribution.png'
    plt.savefig(res_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Residuals Distribution to: {res_path}")
    
    # 7. Generate Future Forecast
    last_seq = X_test[-1]
    future_forecast = forecast_future(
        model=model,
        last_sequence=last_seq,
        target_scaler=prep['target_scaler'],
        feature_scaler=prep['scaler'],
        steps=14,
        n_features=X_train.shape[2]
    )

    # 8. Save Model and Metadata Bundle
    model_save_path = 'saved_models/air_quality_lstm.keras'
    model.save(model_save_path)
    print(f"[OK] Saved trained LSTM model to: {model_save_path}")

    import pickle
    results_bundle = {
        'history': history.history,
        'metrics': metrics,
        'actuals': actuals,
        'preds': preds,
        'future_forecast': future_forecast,
        'target_col': target_col,
        'selected_features': feature_cols,
        'lookback': lookback,
        'prep': prep
    }
    with open('saved_models/model_results.pkl', 'wb') as f:
        pickle.dump(results_bundle, f)
    print("[OK] Saved training results bundle to: saved_models/model_results.pkl")
    print("\nTraining and visualization generation completed successfully!")

if __name__ == '__main__':
    main()
