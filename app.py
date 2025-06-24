import streamlit as st
from fastai.tabular.all import *
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return load_learner('stock_direction_model.pkl')

model = load_model()

# UI Elements
st.title('Stock Direction Predictor')
ticker = st.text_input('Enter stock ticker (e.g., AAPL):', 'AAPL')

if st.button('Predict'):
    try:
        # Get data
        df = yf.download(ticker, period='60d')
        df.reset_index(inplace=True)
        df.columns = df.columns.get_level_values(0)
        
        # Calculate indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df.fillna(method='ffill', inplace=True)
        
        # Prepare for prediction
        latest = df.iloc[-1]
        cont_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_10', 'SMA_50']
        pred_data = pd.DataFrame({col: [latest[col]] for col in cont_names})
        
        # Predict
        pred, _, probs = model.predict(pred_data.iloc[0])
        
        # Display results
        st.success(f"Prediction for {ticker}: {'↑ UP' if pred == '1' else '↓ DOWN'}")
        st.metric("Probability UP", f"{probs[1].item():.2%}")
        st.metric("Current Price", f"${latest['Close']:.2f}")
        
        # Show chart
        st.line_chart(df.set_index('Date')['Close'])
        
    except Exception as e:
        st.error(f"Error: {str(e)}")