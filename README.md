# 6. LSTM – Air Quality Prediction

An end-to-end Deep Learning application using **Long Short-Term Memory (LSTM)** neural networks for **Air Quality Prediction** with an interactive **Streamlit** frontend featuring real-time training, accuracy and loss visualizations, and an interactive prediction generator.

---

## 📌 Project Overview

- **Algorithm**: Stacked 2-Layer LSTM with Dropout Regularization.
- **Task**: Time-Series Forecasting for Air Quality Index (AQI) based on atmospheric pollutant and weather metrics (PM2.5, PM10, NO2, CO, Temperature, Humidity).
- **Frontend**: Clean, simple Streamlit web application.
- **Key Features**:
  - Live dataset overview (`air_quality_data.csv`).
  - Evaluation metrics: $R^2$ Accuracy Score, Loss (MSE), RMSE, and MAE.
  - Accuracy and Loss visualization curves (Train vs Validation).
  - Actual vs Predicted time-series comparison chart.
  - **User Input Prediction**: Input pollutant and weather parameters to generate next-step AQI forecasts and health alerts.
  - Expandable view of the core Python code.

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit web application
├── lstm_model.py           # Core LSTM architecture and sequence generator
├── train_air_quality.py    # Standalone script for training and exporting static plots
├── air_quality_data.csv    # Air quality dataset (PM2.5, PM10, NO2, CO, Temp, Humidity, AQI)
├── data.csv                # Workspace dataset
├── run_project.bat         # 1-Click launcher for Windows
├── requirements.txt        # Python dependencies
├── static/                 # Exported static visualization plots
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── actual_vs_predicted.png
│   └── residuals_distribution.png
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd "d:\rishi project"
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Project

### Method 1: Launch Streamlit Web App (Recommended)
Double-click `run_project.bat` or run:
```bash
python -m streamlit run app.py
```
Open your browser and navigate to:
👉 **`http://localhost:8501`**

### Method 2: Command-Line Training & Plot Generation
To train the model and save the loss/accuracy plots to the `static/` folder:
```bash
python train_air_quality.py
```

---

## 📊 Evaluation & Metrics

- **Loss (MSE)**: Mean Squared Error minimized during training.
- **MAE**: Mean Absolute Error on the test set.
- **RMSE**: Root Mean Squared Error indicating prediction error scale.
- **Accuracy ($R^2$ Score)**: Proportion of variance explained by the LSTM model.

---

## 🧪 Interactive Predictions

In the Streamlit app under **Section 5 ("Enter Inputs to Predict AQI")**:
1. Enter values for **PM2.5**, **PM10**, **NO2**, **CO**, **Temperature**, and **Humidity**.
2. Click **"Generate Prediction"**.
3. View the predicted **AQI** level and corresponding health category (Good, Moderate, Unhealthy, Hazardous).
