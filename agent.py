import os
import json
import datetime
import joblib
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

MODEL_DIR = "models"
SCALER_DIR = "scalers"
META_DIR = "metadata"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


class StockAgent:
    def __init__(self, ticker, lookback=60):
        self.ticker = ticker
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.meta = None

    def fetch_data(self, period="5y"):
        df = yf.download(self.ticker, period=period)
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        df = df[["Close"]].dropna()
        return df

    def preprocess(self, df, fit=True):
        data = df.values.reshape(-1, 1)
        if fit:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = self.scaler.fit_transform(data)
            joblib.dump(self.scaler, os.path.join(SCALER_DIR, f"{self.ticker}_scaler.pkl"))
        else:
            scaled = self.scaler.transform(data)

        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback:i, 0])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, epochs=50, batch_size=32):
        df = self.fetch_data()
        X, y = self.preprocess(df, fit=True)

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model_path = os.path.join(MODEL_DIR, f"{self.ticker}_model.keras")

        model = self.build_model((X_train.shape[1], 1))
        ckpt = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=0)
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size,
                  callbacks=[ckpt, es], verbose=1)

        self.model = model
        self.meta = {"ticker": self.ticker, "lookback": self.lookback,
                     "trained_on": datetime.datetime.now().isoformat()}
        with open(os.path.join(META_DIR, f"{self.ticker}_meta.json"), "w") as f:
            json.dump(self.meta, f)

    def load(self):
        model_path = os.path.join(MODEL_DIR, f"{self.ticker}_model.keras")
        scaler_path = os.path.join(SCALER_DIR, f"{self.ticker}_scaler.pkl")
        meta_path = os.path.join(META_DIR, f"{self.ticker}_meta.json")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            return True
        return False

    def predict(self):
        if not self.load():
            print("No trained model found. Training new model...")
            self.train(epochs=100)

        df = self.fetch_data(period="2y")
        last_window = df[-self.lookback:]["Close"].values.reshape(-1, 1)
        scaled = self.scaler.transform(last_window)

        X_test = np.array([scaled.reshape(-1)])
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        pred_scaled = self.model.predict(X_test)[0][0]
        predicted_close = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        last_close = float(df.iloc[-1]["Close"])
        recent_avg = float(df["Close"].tail(5).mean())  # 5-day moving average

        # ✅ Combined baseline: last_close & recent average
        baseline = (last_close + recent_avg) / 2.0

        diff = predicted_close - baseline
        percent_diff = (diff / baseline) * 100

        # ✅ UP/DOWN with threshold
        if percent_diff > 0.3:
            trend = "UP"
        elif percent_diff < -0.3:
            trend = "DOWN"
        else:
            # agar chhota gap hai to nearest decide karega
            trend = "UP" if predicted_close > last_close else "DOWN"

        return {
            "ticker": self.ticker,
            "last_close": last_close,
            "predicted_close": float(predicted_close),
            "trend": trend
        }
