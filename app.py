from fastapi import FastAPI
import uvicorn
from fastai.tabular.all import *
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import pandas as pd
import numpy as np

app = FastAPI(title="Stock Direction Predictor")

# Load your trained model
try:
    model = load_learner('stock_direction_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Please train and save the model first.")
    print("The model file should be named 'stock_direction_model.pkl'")
    raise

def prepare_prediction_data(df):
    """Prepare a single row of data for prediction in the correct format"""
    # Ensure we have all required columns
    cont_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_10', 'SMA_50']
    
    # Create a new DataFrame with just the columns we need
    pred_df = pd.DataFrame({col: df[col] for col in cont_names})
    
    # FastAI expects specific column order and names
    return pred_df[cont_names]

def get_latest_data(ticker):
    """Fetch and prepare the latest market data"""
    try:
        # Get last 60 days of data
        df = yf.download(ticker, period='60d')
        df.reset_index(inplace=True)
        df.columns = df.columns.get_level_values(0)

        # Calculate indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Fill any NaN values that might occur in indicators
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # Return only the most recent row
        return df.iloc[-1:]
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    try:
        # Get and prepare the data
        latest_data = get_latest_data(ticker)
        pred_data = prepare_prediction_data(latest_data)
        
        # Make prediction
        pred, _, probs = model.predict(pred_data.iloc[0])
        
        return {
            "ticker": ticker,
            "prediction": "Up" if pred == "1" else "Down",
            "probability_up": float(probs[1]),
            "probability_down": float(probs[0]),
            "last_close": float(latest_data['Close'].iloc[0]),
            "timestamp": str(pd.Timestamp.now())
        }
    except Exception as e:
        return {"error": str(e), "details": f"Failed to predict for {ticker}"}

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "endpoints": {
            "predict": "/predict/{ticker}",
            "health": "/"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)