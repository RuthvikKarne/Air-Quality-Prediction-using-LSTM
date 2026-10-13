import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Air Quality Prediction using LSTM", page_icon="🌫️")

st.title("Air Quality Prediction using LSTM")
st.write("Deep Learning model to predict Air Quality Index (AQI) using Long Short-Term Memory (LSTM) neural networks.")

# ---------------------------------------------------------
# 1. Dataset Selection & Display
# ---------------------------------------------------------
st.subheader("1. Air Quality Dataset")

dataset_file = "air_quality_data.csv" if os.path.exists("air_quality_data.csv") else "data.csv"
target_column = "AQI" if os.path.exists("air_quality_data.csv") else "Close"

df = pd.read_csv(dataset_file)
st.write(f"Loaded dataset: `{dataset_file}` ({len(df)} rows)")
st.dataframe(df.head(), width=800)

# ---------------------------------------------------------
# 2. Sidebar Parameters
# ---------------------------------------------------------
st.sidebar.header("Model Settings")
lookback = st.sidebar.slider("Lookback Window (Days)", min_value=7, max_value=30, value=14)
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=50, value=20, step=5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
train_btn = st.sidebar.button("Train Model", type="primary")

# ---------------------------------------------------------
# 3. Model Training Function
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_lstm_model(lookback_days, num_epochs, b_size):
    # Select features
    if "AQI" in df.columns:
        features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'Temperature', 'Humidity']
        target_col = 'AQI'
    else:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = features[0]

    raw_data = df[features].dropna().values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[[target_col]].dropna().values)
    
    # Create Sequences
    X, y = [], []
    for i in range(lookback_days, len(scaled_data)):
        X.append(scaled_data[i - lookback_days:i, :])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # 80/20 Train Test Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(25, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = model.fit(
        X_train, y_train,
        epochs=num_epochs,
        batch_size=b_size,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Predictions
    preds_scaled = model.predict(X_test, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Metrics
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    accuracy_pct = max(0.0, r2 * 100.0)
    
    return {
        'history': history.history,
        'actuals': actuals,
        'preds': preds,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'accuracy_pct': accuracy_pct,
        'target_col': target_col,
        'features': features,
        'model': model,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'last_sequence': X_test[-1],
        'lookback': lookback_days
    }

# Load or Train Model
results = None
if train_btn:
    st.cache_resource.clear()
    with st.spinner("Training LSTM model... please wait"):
        results = train_lstm_model(lookback, epochs, batch_size)
    st.success("Model trained successfully!")
else:
    # Auto-load default training so page shows visualizations immediately
    results = train_lstm_model(lookback, epochs, batch_size)

# ---------------------------------------------------------
# 4. Metrics & Accuracy Display
# ---------------------------------------------------------
st.subheader("2. Model Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (R² Score)", f"{results['accuracy_pct']:.2f}%")
col2.metric("Loss (MSE)", f"{results['mse']:.4f}")
col3.metric("RMSE", f"{results['rmse']:.2f}")
col4.metric("MAE", f"{results['mae']:.2f}")

# ---------------------------------------------------------
# 5. Visualizations (Loss & Accuracy)
# ---------------------------------------------------------
st.subheader("3. Accuracy and Loss Visualization")

history = results['history']
actuals = results['actuals']
preds = results['preds']
t_col = results['target_col']

# Row 1: Loss & Accuracy Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss Curve
ax1.plot(history['loss'], label='Training Loss (MSE)', color='blue', linewidth=2)
ax1.plot(history['val_loss'], label='Validation Loss (MSE)', color='red', linestyle='--', linewidth=2)
ax1.set_title('Model Loss Over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy / MAE Curve
ax2.plot(history['mae'], label='Training MAE', color='green', linewidth=2)
ax2.plot(history['val_mae'], label='Validation MAE', color='orange', linestyle='--', linewidth=2)
ax2.set_title('Model MAE (Accuracy Tracking) Over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Row 2: Prediction vs Actual
st.subheader(f"4. Actual vs Predicted {t_col} Values")

fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(actuals, label='Actual Values', color='blue', linewidth=2)
ax.plot(preds, label='LSTM Predictions', color='red', linestyle='--', linewidth=2)
ax.set_title(f'Actual vs Predicted {t_col}')
ax.set_xlabel('Test Sample Index (Days)')
ax.set_ylabel(f'{t_col} Level')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)

# ---------------------------------------------------------
# 6. User Input & Prediction Generation
# ---------------------------------------------------------
st.subheader(f"5. Enter Inputs to Predict {t_col}")
st.write(f"Enter the pollutant and atmospheric parameters below to generate the predicted {t_col}:")

input_features = [f for f in results['features'] if f != t_col]
if not input_features:
    input_features = results['features']

# Arrange inputs in a 3-column grid
cols = st.columns(3)
user_values = {}

for idx, feat in enumerate(input_features):
    with cols[idx % 3]:
        default_val = float(df[feat].mean()) if feat in df.columns else 50.0
        min_val = float(df[feat].min()) if feat in df.columns else 0.0
        max_val = float(df[feat].max()) if feat in df.columns else 500.0
        user_values[feat] = st.number_input(
            f"{feat}",
            value=round(default_val, 2),
            min_value=round(min_val, 2),
            max_value=round(max_val, 2)
        )

if st.button("Generate Prediction", type="primary"):
    # Build complete row matching model features
    row_values = []
    for feat in results['features']:
        if feat == t_col:
            row_values.append(float(df[t_col].mean()))
        else:
            row_values.append(user_values.get(feat, float(df[feat].mean())))
            
    # Scale input row
    row_array = np.array(row_values).reshape(1, -1)
    scaled_row = results['scaler'].transform(row_array)[0]
    
    # Construct input sequence using last sequence + user input
    input_seq = np.copy(results['last_sequence'])
    input_seq[-1] = scaled_row
    input_seq_batch = input_seq.reshape(1, results['lookback'], -1)
    
    # Model predict
    pred_scaled = results['model'].predict(input_seq_batch, verbose=0)[0, 0]
    pred_val = float(results['target_scaler'].inverse_transform([[pred_scaled]])[0, 0])
    
    # Display Output
    st.markdown("---")
    st.success(f"### 🎯 Predicted {t_col}: **{pred_val:.2f}**")
    
    # If predicting AQI, show Air Quality category
    if t_col == "AQI":
        if pred_val <= 50:
            st.info("🟢 **Air Quality: Good** (Satisfactory air quality, little or no health risk.)")
        elif pred_val <= 100:
            st.warning("🟡 **Air Quality: Moderate** (Acceptable; sensitive individuals may experience minor symptoms.)")
        elif pred_val <= 150:
            st.warning("🟠 **Air Quality: Unhealthy for Sensitive Groups** (Children and elderly may experience irritation.)")
        elif pred_val <= 200:
            st.error("🔴 **Air Quality: Unhealthy** (Everyone may begin to experience health effects.)")
        elif pred_val <= 300:
            st.error("🟣 **Air Quality: Very Unhealthy** (Health alert: everyone may experience serious effects.)")
        else:
            st.error("🟤 **Air Quality: Hazardous** (Emergency health conditions: entire population affected.)")

# ---------------------------------------------------------
# 7. Python Code Display
# ---------------------------------------------------------
with st.expander("📄 View LSTM Python Code"):
    st.code('''# 6. LSTM – Air Quality Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# 1. Load Data
df = pd.read_csv('air_quality_data.csv')
features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'Temperature', 'Humidity']
data = df[features].values

# 2. Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Sliding Window Sequences
lookback = 14
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, :])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. LSTM Architecture
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(25, return_sequences=False, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 6. Evaluate
preds = model.predict(X_test)

# 7. Plot Loss and Accuracy
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
''', language='python')
