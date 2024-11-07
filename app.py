# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os
import tempfile
from io import BytesIO

# Data and API libraries
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from fredapi import Fred
from openai import OpenAI

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go

# Optimization libraries
from scipy.optimize import minimize
from tqdm import tqdm

# Audio processing libraries
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS

# Add this with your other session state initializations
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Set page configuration
st.set_page_config(
    page_title="ML-Enhanced Portfolio App",
    page_icon="📈",
    layout="wide"
)

# Session state initialization
if 'data_gathering_complete' not in st.session_state:
    st.session_state.data_gathering_complete = False
if 'data_merging_complete' not in st.session_state:
    st.session_state.data_merging_complete = False


# Fetch API keys from secrets
OPENAI_API_KEY = st.secrets["api_keys"]["openai_api_key"]
FRED_API_KEY = st.secrets["api_keys"]["fred_api_key"]
ADMIN_PASSWORD = st.secrets["admin"]["password"]


def voice_ai_assistant():
    st.title("🎙️ Voice-Activated AI Portfolio Assistant")

    # Initialize session state variables
    if 'voice_messages' not in st.session_state:
        st.session_state.voice_messages = []

    # Use the API key from secrets
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return

    # Main interface layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        ### Voice AI Financial Advisor
        Ask questions about:
        - Portfolio analysis
        - Investment strategies
        - Market insights
        - Risk management
        - Financial planning
        """)

    # Audio recording
    with col2:
        # Using the audio_recorder component
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            pause_threshold=2.0,
            sample_rate=44100
        )

        if audio_bytes:
            try:
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name

                # Process the audio file
                with open(temp_audio_path, 'rb') as audio_file:
                    st.info("🔄 Transcribing audio...")
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )

                transcribed_text = transcription.text
                st.success(f"🎯 Transcribed: {transcribed_text}")

                # Add user message to chat history
                st.session_state.voice_messages.append({
                    "role": "user",
                    "content": transcribed_text,
                    "type": "voice"
                })

                # Get AI response
                st.info("🤖 Getting AI response...")
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an AI financial advisor. Provide clear, 
                            concise advice about investments, portfolio management, and 
                            financial planning. Keep responses focused and actionable."""
                        },
                        {
                            "role": "user",
                            "content": transcribed_text
                        }
                    ]
                )

                ai_response = response.choices[0].message.content

                # Add AI response to chat history
                st.session_state.voice_messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "type": "voice"
                })

                # Convert to speech
                st.info("🔊 Converting response to speech...")
                tts = gTTS(text=ai_response, lang='en')
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp.getvalue(), format='audio/mp3')

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.error("Please try recording again")
            finally:
                # Cleanup
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

    # Display conversation history
    st.markdown("### Conversation History")
    for msg in st.session_state.voice_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            st.caption("🎤 Voice message" if msg["role"] == "user" else "🔊 Voice response")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.voice_messages = []
        st.experimental_rerun()

    # Help section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Click the recording button (it will turn red when recording)
        2. Allow microphone access if prompted
        3. Speak your question clearly
        4. Click the button again to stop recording
        5. Wait for the AI to process and respond
        6. Listen to the response and view the transcript
        
        **Tips for best results:**
        - Use a clear, quiet environment
        - Speak at a normal pace
        - Keep questions focused and specific
        - Make sure your browser has microphone permissions
        - Record for at least 2-3 seconds
        """)


def ai_portfolio_assistant():
    st.title("🤖 AI Portfolio Assistant")

    # Use the API key from secrets
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize chat interface
        st.markdown("""
        ### Chat with your AI Financial Advisor
        Get personalized portfolio advice, market insights, and answers to your financial questions.
        """)

        # Initialize message history if empty
        if len(st.session_state.messages) == 0:
            system_msg = {
                "role": "system",
                "content": """You are an AI financial advisor designed to assist users with portfolio management, financial planning, and investment strategies. Provide accurate, concise, and insightful advice based on user input. Use plain language to explain complex financial concepts, and ensure all recommendations comply with best financial practices. Always prioritize risk management, long-term growth, and diversification in your suggestions. When unsure, recommend consulting a licensed financial advisor."""
            }
            st.session_state.messages.append(system_msg)

            # Add welcome message
            welcome_msg = {
                "role": "assistant",
                "content": """Hello! I'm your AI Financial Advisor. I can help you with:
- Portfolio analysis and recommendations
- Investment strategy development
- Risk management advice
- Market insights and analysis
- Basic financial planning

What would you like to discuss today?"""
            }
            st.session_state.messages.append(welcome_msg)

        # Display chat history
        for message in st.session_state.messages[1:]:  # Skip system message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask your financial question..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    response = client.chat.completions.create(
                        model="gpt-4",  # Using GPT-4 for better financial advice
                        messages=[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state.messages],
                        stream=True
                    )

                    # Stream the response
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "Rate limit" in str(e):
                        st.warning("Rate limit reached. Please wait a moment before sending another message.")
                    elif "Incorrect API key" in str(e):
                        st.warning("Invalid API key. Please check your OpenAI API key and try again.")
                    else:
                        st.warning("An error occurred. Please try again later.")

    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")

# Admin Dashboard Updates
# Use `FRED_API_KEY` where needed for FRED data fetching.

# The rest of the functions will remain unchanged except for the use of these API keys internally where applicable.

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.findAll('td')[0].text.strip().replace('.', '-') for row in table.findAll('tr')[1:]]
    return tickers

def fetch_stock_data(tickers, start_date, end_date, progress_bar):
    """Fetch historical stock data."""
    all_data = []
    for i, ticker in enumerate(tickers):
        try:
            progress_bar.progress((i + 1) / len(tickers), f"Fetching {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                hist.reset_index(inplace=True)
                hist['Ticker'] = ticker
                all_data.append(hist[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        st.error("No stock data fetched. Please check your internet connection or the ticker symbols.")
        return pd.DataFrame()

def fetch_fred_data(fred_api_key, start_date, end_date, progress_bar):
    """Fetch FRED data."""
    fred = Fred(api_key=fred_api_key)
    series = {
        'Consumer_Sentiment_Index': 'UMCSENT',
        'VIX': 'VIXCLS',
        'Baa_Yield': 'DBAA',
        'Aaa_Yield': 'DAAA',
        'Ten_Year_Yield': 'DGS10',
        'Two_Year_Yield': 'DGS2',
        'Financial_Conditions_Index': 'NFCI',
        'Economic_Policy_Uncertainty': 'USEPUINDXD'
    }
    
    combined_data = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date)})
    for i, (name, series_id) in enumerate(series.items()):
        try:
            progress_bar.progress((i + 1) / len(series), f"Fetching {name}")
            data = fred.get_series(series_id, start_date, end_date).reset_index()
            data.columns = ['Date', name]
            combined_data = pd.merge(combined_data, data, on='Date', how='outer')
        except Exception as e:
            st.error(f"Error fetching {name}: {e}")
    
    return combined_data

def merge_financial_data_with_progress(stock_data_df, risk_metrics_df, progress_bar):
    """Merge stock and FRED data with progress tracking."""
    try:
        progress_bar.progress(0.1, text="Converting dates to datetime...")
    
        stock_data_df['Date'] = pd.to_datetime(stock_data_df['Date'].astype(str).str.split().str[0])
        risk_metrics_df['Date'] = pd.to_datetime(risk_metrics_df['Date'])
    
        stock_data_df['Year'] = stock_data_df['Date'].dt.year
        stock_data_df['Month'] = stock_data_df['Date'].dt.month
        risk_metrics_df['Year'] = risk_metrics_df['Date'].dt.year
        risk_metrics_df['Month'] = risk_metrics_df['Date'].dt.month
    
        progress_bar.progress(0.4, text="Aggregating risk metrics data...")
    
        fred_columns = [
            'Year', 'Month', 'Consumer_Sentiment_Index', 'VIX', 'Baa_Yield',
            'Aaa_Yield', 'Ten_Year_Yield', 'Two_Year_Yield',
            'Financial_Conditions_Index', 'Economic_Policy_Uncertainty'
        ]
        risk_metrics_df = risk_metrics_df[fred_columns].groupby(['Year', 'Month']).first().reset_index()
    
        progress_bar.progress(0.7, text="Merging stock and risk metrics data...")
    
        merged_data = pd.merge(
            stock_data_df,
            risk_metrics_df,
            on=['Year', 'Month'],
            how='left'
        )
        merged_data = merged_data.drop(['Year', 'Month'], axis=1)
    
        progress_bar.progress(1.0, text="Sorting and cleaning merged data...")
        return merged_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error during data merging: {str(e)}")
        raise

def calculate_technical_indicators_admin(df):
    """Calculate technical indicators for each stock."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    tickers = df['Ticker'].unique()
    for ticker in tqdm(tickers, desc="Processing Tickers"):
        mask = df['Ticker'] == ticker
        df.loc[mask, 'Daily_Return'] = df.loc[mask, 'Close'].pct_change()
        df.loc[mask, 'MA_7'] = df.loc[mask, 'Close'].rolling(window=7).mean()
        df.loc[mask, 'MA_30'] = df.loc[mask, 'Close'].rolling(window=30).mean()
        df.loc[mask, 'Volatility_7'] = df.loc[mask, 'Daily_Return'].rolling(window=7).std()
        df.loc[mask, 'Volatility_30'] = df.loc[mask, 'Daily_Return'].rolling(window=30).std()
        df.loc[mask, 'Volume_MA_7'] = df.loc[mask, 'Volume'].rolling(window=7).mean()
        df.loc[mask, 'Volume_Ratio'] = df.loc[mask, 'Volume'] / df.loc[mask, 'Volume_MA_7']
        df.loc[mask, 'Price_Range'] = (df.loc[mask, 'High'] - df.loc[mask, 'Low']) / df.loc[mask, 'Open']
        for lag in range(1, 6):
            df.loc[mask, f'Close_Lag_{lag}'] = df.loc[mask, 'Close'].shift(lag)
            df.loc[mask, f'Volatility_30_Lag_{lag}'] = df.loc[mask, 'Volatility_30'].shift(lag)
            df.loc[mask, f'Volume_Lag_{lag}'] = df.loc[mask, 'Volume'].shift(lag)
        df.loc[mask, 'Volatility_Interaction'] = df.loc[mask, 'Volatility_30'] * df.loc[mask, 'VIX']
        df.loc[mask, 'Yield_Spread'] = df.loc[mask, 'Ten_Year_Yield'] - df.loc[mask, 'Two_Year_Yield']
        df.loc[mask, 'Rolling_Percentile_25'] = df.loc[mask, 'Close'].rolling(30).quantile(0.25)
        df.loc[mask, 'Rolling_Percentile_75'] = df.loc[mask, 'Close'].rolling(30).quantile(0.75)
        df.loc[mask, 'Rate_of_Change'] = df.loc[mask, 'Close'].pct_change(periods=5)
        df.loc[mask, 'EMA_10'] = df.loc[mask, 'Close'].ewm(span=10, adjust=False).mean()
    return df

def train_model(df):
    """Train a Random Forest model with optimized features."""
    df = calculate_technical_indicators_admin(df)
    features = [
        'Daily_Return', 'MA_7', 'MA_30', 'Volatility_7', 'Volatility_30',
        'Volume_MA_7', 'Volume_Ratio', 'Price_Range',
        'Volatility_Interaction', 'Yield_Spread',
        'Rolling_Percentile_25', 'Rolling_Percentile_75', 'Rate_of_Change', 'EMA_10',
        'VIX', 'Consumer_Sentiment_Index', 'Financial_Conditions_Index', 'Economic_Policy_Uncertainty',
        'Baa_Yield', 'Aaa_Yield', 'Ten_Year_Yield', 'Two_Year_Yield'
    ]
    df['Future_Volatility'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.rolling(30).std().shift(-30)
    )
    df_clean = df.dropna(subset=features + ['Future_Volatility'])
    X = df_clean[features]
    y = df_clean['Future_Volatility']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(
        n_estimators=400, max_depth=25, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"Model Performance: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    # Save model components
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    with open('model_features.txt', 'w') as f:
        f.write('\n'.join(features))
    st.success("Model training completed and model components saved, including model_features.txt!")

def portfolio_model_admin_dashboard():
    st.title("🛠️ Portfolio Model Admin Dashboard")

    # Password protection
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.subheader("🔒 Admin Login")
        password = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Incorrect password.")
        return  # Stop execution until authenticated

    # Use FRED API Key from secrets
    fred_api_key = FRED_API_KEY

    st.sidebar.header("Settings")
    tabs = st.tabs(["Data Gathering", "Data Merging", "Model Training"])

    with tabs[0]:
        st.header("Step 1: Data Gathering")
        start_date = st.date_input("Start Date", dt.datetime(2022, 1, 1))
        end_date = st.date_input("End Date", dt.datetime.today())
        num_stocks = st.slider("Number of S&P 500 stocks", 10, 100, 50, 10)
        if st.button("Start Data Gathering"):
            try:
                tickers = get_sp500_tickers()[:num_stocks] + ['GLD', 'SLV']
                stock_progress = st.progress(0)
                fred_progress = st.progress(0)
                
                # Fetch stock data
                stock_data = fetch_stock_data(tickers, start_date, end_date, stock_progress)
                if not stock_data.empty:
                    stock_data.to_csv('sp500_data.csv', index=False)
                else:
                    st.error("Stock data is empty. Data gathering aborted.")
                    return
                
                # Fetch FRED data
                fred_data = fetch_fred_data(fred_api_key, start_date, end_date, fred_progress)
                if not fred_data.empty:
                    fred_data.to_csv('fred_data.csv', index=False)
                else:
                    st.error("FRED data is empty. Data gathering aborted.")
                    return
                
                st.session_state.data_gathering_complete = True
                st.success("Data gathering completed!")
            except Exception as e:
                st.error(f"Error during data gathering: {e}")

    with tabs[1]:
        st.header("Step 2: Data Merging")
        if st.button("Start Data Merging"):
            try:
                stock_data = pd.read_csv('sp500_data.csv')
                fred_data = pd.read_csv('fred_data.csv')
                progress_bar = st.progress(0)
                merged_data = merge_financial_data_with_progress(stock_data, fred_data, progress_bar)
                merged_data.to_csv('merged_financial_data.csv', index=False)
                st.session_state.data_merging_complete = True
                st.success("Data merging completed!")
            except Exception as e:
                st.error(f"Error during data merging: {e}")

    with tabs[2]:
        st.header("Step 3: Model Training")
        if st.button("Start Model Training"):
            try:
                df = pd.read_csv('merged_financial_data.csv')
                train_model(df)
            except Exception as e:
                st.error(f"Error during model training: {e}")


# ------------------ Portfolio Risk Analyzer Functions ------------------

@st.cache_resource
def load_ml_components():
    """Load trained model, scaler, and features"""
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('model_features.txt', 'r') as f:
            features = f.read().splitlines()
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading ML components: {e}")
        return None, None, None

def calculate_technical_indicators(stock_data):
    """Calculate technical indicators needed for the model"""
    df = stock_data.copy()
    
    # Basic indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']
    df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
    
    # Lagged features
    for lag in range(1, 6):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volatility_30_Lag_{lag}'] = df['Volatility_30'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Additional indicators
    df['Rate_of_Change'] = df['Close'].pct_change(periods=5)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Rolling_Percentile_25'] = df['Close'].rolling(30).quantile(0.25)
    df['Rolling_Percentile_75'] = df['Close'].rolling(30).quantile(0.75)
    
    # Interaction features
    if 'VIX' in df.columns:
        df['Volatility_Interaction'] = df['Volatility_30'] * df['VIX']
    else:
        df['Volatility_Interaction'] = df['Volatility_30']  # Fallback if VIX not available
        
    if 'Ten_Year_Yield' in df.columns and 'Two_Year_Yield' in df.columns:
        df['Yield_Spread'] = df['Ten_Year_Yield'] - df['Two_Year_Yield']
    else:
        df['Yield_Spread'] = 0  # Fallback if yield data not available
        
    return df

def predict_volatility(data, model, scaler, features):
    """Predict future volatility using the trained model"""
    try:
        # Prepare features
        X = data[features].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        predicted_volatility = model.predict(X_scaled)
        
        return predicted_volatility[-1]  # Return the latest prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def calculate_risk_status(volatility, drawdown):
    """Calculate risk status based on volatility and drawdown"""
    vol = volatility
    dd = abs(drawdown)
    
    if vol < 0.15 and dd < 0.20:
        return "Low Risk", "green", """
        - Conservative investment profile
        - Suitable for risk-averse investors
        - Typically seen in blue-chip stocks or defensive sectors
        """
    elif vol < 0.25 and dd < 0.30:
        return "Moderate Risk", "yellow", """
        - Balanced risk-reward profile
        - Suitable for most long-term investors
        - Common among established growth companies
        """
    elif vol < 0.35 and dd < 0.40:
        return "High Risk", "orange", """
        - Aggressive investment profile
        - Suitable for risk-tolerant investors
        - Typical of growth stocks and tech sector
        """
    else:
        return "Very High Risk", "red", """
        - Highly aggressive investment profile
        - Suitable only for very risk-tolerant investors
        - Characteristic of volatile tech stocks and emerging companies
        """

def explain_metrics(investment_amount, volatility, drawdown, predicted_volatility=None):
    """Generate detailed explanation of risk metrics"""
    vol_amount = investment_amount * volatility
    drawdown_amount = investment_amount * abs(drawdown)
    
    volatility_explanation = f"""
    📊 **Annualized Volatility**:
    - Historical: {volatility:.2%}
    {f'- Predicted: {predicted_volatility:.2%}' if predicted_volatility is not None else ''}
    
    For your **${investment_amount:,.2f}** investment:
    - Typical yearly value fluctuation: ±**${vol_amount:,.2f}**
    - 68% of the time, your investment could range between:
        - High: **${(investment_amount + vol_amount):,.2f}**
        - Low: **${(investment_amount - vol_amount):,.2f}**
    - 95% of the time (2 standard deviations), range between:
        - High: **${(investment_amount + 2*vol_amount):,.2f}**
        - Low: **${(investment_amount - 2*vol_amount):,.2f}**
    """
    
    drawdown_explanation = f"""
    📉 **Maximum Drawdown ({drawdown:.2%})**:
    
    Worst historical decline scenario:
    - Starting value: **${investment_amount:,.2f}**
    - Lowest value: **${(investment_amount * (1 + drawdown)):,.2f}**
    - Maximum loss: **${drawdown_amount:,.2f}**
    
    This means if you had invested **${investment_amount:,.2f}** at the peak,
    it would have temporarily declined by **${drawdown_amount:,.2f}** at its worst point.
    """
    
    return volatility_explanation, drawdown_explanation

def calculate_stock_metrics(stock_data, model, scaler, features):
    """Calculate comprehensive metrics for individual stock"""
    returns = stock_data['Close'].pct_change().dropna()
    
    # Calculate historical metrics
    volatility_daily = returns.std()
    volatility_annualized = volatility_daily * np.sqrt(252)
    volatility_monthly = volatility_daily * np.sqrt(21)
    annualized_return = (1 + returns.mean()) ** 252 - 1
    
    # Calculate drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate Sharpe Ratio (assuming 2% risk-free rate)
    sharpe_ratio = (annualized_return - 0.02) / volatility_annualized
    
    # New metrics from updated model
    rate_of_change = stock_data['Close'].pct_change(periods=5).mean()
    price_range_avg = ((stock_data['High'] - stock_data['Low']) / stock_data['Open']).mean()
    
    # Prepare data for ML prediction
    prepared_data = calculate_technical_indicators(stock_data)
    predicted_vol_daily = predict_volatility(prepared_data, model, scaler, features)
    if predicted_vol_daily is not None:
        predicted_vol_annualized = predicted_vol_daily * np.sqrt(252)
        predicted_vol_monthly = predicted_vol_daily * np.sqrt(21)
    else:
        predicted_vol_annualized = None
        predicted_vol_monthly = None
    
    return {
        'Annual Return': f"{annualized_return:.2%}",
        'Historical Volatility (Annualized)': f"{volatility_annualized:.2%}",
        'Historical Volatility (Monthly)': f"{volatility_monthly:.2%}",
        'Predicted Volatility (Annualized)': f"{predicted_vol_annualized:.2%}" if predicted_vol_annualized is not None else "N/A",
        'Predicted Volatility (Monthly)': f"{predicted_vol_monthly:.2%}" if predicted_vol_monthly is not None else "N/A",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        '5-Day Rate of Change': f"{rate_of_change:.2%}",
        'Avg Price Range': f"{price_range_avg:.2%}"
    }

def create_risk_gauge(volatility, predicted_volatility, risk_level, risk_color):
    """Create an enhanced risk gauge visualization"""
    fig = go.Figure()
    
    # Historical volatility gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=volatility * 100,
        domain={'x': [0, 0.45], 'y': [0, 1]},
        title={'text': f"Historical Volatility"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 15], 'color': "green"},
                {'range': [15, 25], 'color': "yellow"},
                {'range': [25, 35], 'color': "orange"},
                {'range': [35, 50], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': volatility * 100
            }
        }
    ))
    
    # Predicted volatility gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predicted_volatility * 100,
        domain={'x': [0.55, 1], 'y': [0, 1]},
        title={'text': f"Predicted Volatility"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 15], 'color': "green"},
                {'range': [15, 25], 'color': "yellow"},
                {'range': [25, 35], 'color': "orange"},
                {'range': [35, 50], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_volatility * 100
            }
        }
    ))
    
    fig.update_layout(
        title=f"Portfolio Risk Level: {risk_level}",
        height=400
    )
    
    return fig

def create_correlation_heatmap(returns_data):
    """Create correlation heatmap"""
    corr_matrix = returns_data.corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Stock", y="Stock", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale="RdBu",
                    aspect="auto")
    fig.update_layout(title="Stock Correlation Heatmap")
    return fig

def create_metrics_dashboard(returns_data, investment_amounts, predicted_volatilities=None):
    """Create comprehensive risk metrics dashboard"""
    total_investment = sum(investment_amounts.values())
    weights = {k: v / total_investment for k, v in investment_amounts.items()}
    
    portfolio_returns = returns_data.dot(pd.Series(weights))
    
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate
    
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    metrics = {
        'Annual Return': f"{annualized_return:.2%}",
        'Historical Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Daily VaR (95%)': f"{np.percentile(portfolio_returns, 5):.2%}",
        'Daily VaR (99%)': f"{np.percentile(portfolio_returns, 1):.2%}",
        'Sortino Ratio': f"{(annualized_return - 0.02) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)):.2f}"
    }
    
    # Add predicted volatilities if available
    if predicted_volatilities is not None:
        weights_list = list(weights.values())
        weighted_pred_vol_annualized = sum(w * v * np.sqrt(252) for w, v in zip(weights_list, predicted_volatilities) if v is not None)
        weighted_pred_vol_monthly = sum(w * v * np.sqrt(21) for w, v in zip(weights_list, predicted_volatilities) if v is not None)
        metrics['Predicted Volatility (Annualized)'] = f"{weighted_pred_vol_annualized:.2%}"
        metrics['Predicted Volatility (Monthly)'] = f"{weighted_pred_vol_monthly:.2%}"
    
    return metrics, volatility, max_drawdown

def get_optimal_portfolio(data, target_risk_min, target_risk_max, num_stocks, model, scaler, features):
    """Get optimal portfolio based on target risk range and ML predictions"""
    try:
        # Calculate stock metrics including ML predictions
        stock_metrics = []
        for ticker in data['Ticker'].unique():
            stock_data = data[data['Ticker'] == ticker]
            if len(stock_data) > 126:  # Ensure sufficient data
                prepared_data = calculate_technical_indicators(stock_data)
                predicted_vol_daily = predict_volatility(prepared_data, model, scaler, features)
                
                # Skip if prediction failed
                if predicted_vol_daily is None:
                    continue
                
                # Calculate predicted monthly volatility
                predicted_vol_monthly = predicted_vol_daily * np.sqrt(21)
                
                # Filter stocks based on predicted monthly volatility being within target range
                if not (target_risk_min <= predicted_vol_monthly <= target_risk_max):
                    continue
                
                # Calculate additional metrics for scoring
                returns = stock_data['Close'].pct_change().dropna()
                sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
                
                stock_metrics.append({
                    'Ticker': ticker,
                    'predicted_volatility_monthly': predicted_vol_monthly,
                    'market_cap': stock_data['Close'].mean() * stock_data['Volume'].mean(),
                    'sharpe_ratio': sharpe
                })
        
        if not stock_metrics:
            st.error("No stocks found matching the target volatility range.")
            return None
        
        stock_metrics_df = pd.DataFrame(stock_metrics)
        
        # Enhanced scoring system
        mid_target_risk = (target_risk_min + target_risk_max) / 2
        stock_metrics_df['vol_score'] = abs(stock_metrics_df['predicted_volatility_monthly'] - mid_target_risk)
        stock_metrics_df['size_score'] = stock_metrics_df['market_cap'].rank(pct=True)
        stock_metrics_df['sharpe_score'] = stock_metrics_df['sharpe_ratio'].rank(pct=True)
        
        # Combine scores with weights
        stock_metrics_df['total_score'] = (
            stock_metrics_df['vol_score'] * -1 +    # Lower vol difference is better
            stock_metrics_df['size_score'] * 0.3 +  # Some weight to market cap
            stock_metrics_df['sharpe_score'] * 0.7  # Higher weight to Sharpe ratio
        )
        
        # Get top candidates
        top_candidates = stock_metrics_df.nlargest(num_stocks * 3, 'total_score')
        
        # Randomly select from top candidates
        selected_stocks = np.random.choice(
            top_candidates['Ticker'].values, 
            size=min(num_stocks, len(top_candidates)),
            replace=False
        )
        
        return list(selected_stocks)
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        return None

def suggest_portfolio(data, returns_data, target_risk, model, scaler, features):
    """Generate optimal weights considering ML predictions"""
    returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    
    # Get predicted volatilities for each stock
    predicted_vols = []
    tickers = returns_data.columns
    for stock in tickers:
        stock_data = data[data['Ticker'] == stock]
        prepared_data = calculate_technical_indicators(stock_data)
        pred_vol = predict_volatility(prepared_data, model, scaler, features)
        if pred_vol is None:
            pred_vol = returns_data[stock].std() * np.sqrt(252)  # Fallback to historical volatility
        predicted_vols.append(pred_vol)
    
    def portfolio_stats(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        pred_portfolio_vol = np.sqrt(sum((w * v) ** 2 for w, v in zip(weights, predicted_vols)))
        sharpe = (portfolio_return - 0.02) / portfolio_vol
        return portfolio_vol, pred_portfolio_vol, sharpe
    
    def objective(weights):
        try:
            portfolio_vol, pred_portfolio_vol, sharpe = portfolio_stats(weights)
            # Combined score considering both historical and predicted volatility
            combined_vol = (portfolio_vol * 0.3 + pred_portfolio_vol * 0.7)
            return abs(combined_vol - target_risk) - 0.1 * sharpe
        except:
            return 1e10  # Return large number if calculation fails
    
    n_stocks = len(returns)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda x: x - 0.05}      # Minimum 5% per stock
    ]
    
    bounds = tuple((0.05, 0.4) for _ in range(n_stocks))  # Min 5%, Max 40% per stock
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple random starting points
    for _ in range(10):  # Increased from 5 to 10 attempts
        try:
            # Generate random weights that sum to 1
            weights = np.random.random(n_stocks)
            weights = weights / np.sum(weights)
            
            result = minimize(
                objective,
                weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}  # Increased max iterations
            )
            
            if result.success and result.fun < best_score:
                best_result = result
                best_score = result.fun
        except Exception:
            continue
    
    if best_result is None:
        # If optimization fails, return equal weights as fallback
        equal_weights = {ticker: 1.0/n_stocks for ticker in tickers}
        return equal_weights
    
    # Return optimized weights
    return {tickers[i]: best_result.x[i] for i in range(n_stocks)}

def portfolio_risk_analyzer():
    st.title("📈 ML-Enhanced Portfolio Risk Analyzer")
    st.markdown("""
    This advanced tool provides detailed risk analysis and portfolio suggestions using machine learning predictions.
    The model considers multiple technical indicators, market conditions, and historical patterns to forecast volatility.
    """)
    
    # Load ML components
    model, scaler, features = load_ml_components()
    if None in (model, scaler, features):
        st.error("Failed to load ML components. Some features may be unavailable.")
        return
    
    # Load data
    try:
        data = pd.read_csv('merged_financial_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        available_tickers = sorted(data['Ticker'].unique())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Analysis type selection with more detailed descriptions
    st.sidebar.header("📊 Analysis Options")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Manual Portfolio", "Suggested Portfolio"],
        help="""
        Manual Portfolio: Build and analyze your own portfolio
        Suggested Portfolio: Get ML-optimized portfolio suggestions based on your risk preference
        """
    )
    
    if analysis_type == "Manual Portfolio":
        st.sidebar.header("Portfolio Selection")
        
        selected_stocks = st.sidebar.multiselect(
            "Select stocks for your portfolio:",
            options=available_tickers,
            help="Choose stocks to analyze"
        )
    
        if selected_stocks:
            latest_data = data[data['Ticker'].isin(selected_stocks)].groupby('Ticker').last()
            
            st.subheader("💰 Investment Details")
            investment_amounts = {}
            stock_predictions = {}
            
            # Calculate predictions and show current metrics for each stock
            for stock in selected_stocks:
                stock_data = data[data['Ticker'] == stock]
                prepared_data = calculate_technical_indicators(stock_data)
                pred_vol_daily = predict_volatility(prepared_data, model, scaler, features)
                
                if pred_vol_daily is not None:
                    pred_vol_annualized = pred_vol_daily * np.sqrt(252)
                    pred_vol_monthly = pred_vol_daily * np.sqrt(21)
                    stock_predictions[stock] = {
                        'daily': pred_vol_daily,
                        'annualized': pred_vol_annualized,
                        'monthly': pred_vol_monthly
                    }
                else:
                    stock_predictions[stock] = None
                
                latest_price = latest_data.loc[stock, 'Close']
                col1, col2 = st.columns([2, 1])
                with col1:
                    shares = st.number_input(
                        f"{stock} shares (${latest_price:.2f}/share)",
                        min_value=0,
                        value=100,
                        step=1
                    )
                with col2:
                    st.metric(
                        "Predicted Monthly Volatility",
                        f"{stock_predictions[stock]['monthly']:.1%}" if stock_predictions[stock] else "N/A"
                    )
                investment_amounts[stock] = shares * latest_price
    
            if st.button("🔍 Analyze Portfolio Risk"):
                with st.spinner("Analyzing portfolio risk and generating insights..."):
                    # Calculate portfolio metrics
                    total_investment = sum(investment_amounts.values())
                    weights = {stock: amt / total_investment for stock, amt in investment_amounts.items()}
                    
                    selected_data = data[data['Ticker'].isin(selected_stocks)].copy()
                    returns_data = selected_data.pivot(
                        index='Date',
                        columns='Ticker',
                        values='Close'
                    ).pct_change()
                    
                    # Get metrics with both historical and predicted volatilities
                    predicted_volatilities = [
                        stock_predictions[s]['daily'] if stock_predictions[s] is not None else None 
                        for s in selected_stocks
                    ]
                    metrics, volatility, max_drawdown = create_metrics_dashboard(
                        returns_data, 
                        investment_amounts,
                        predicted_volatilities
                    )
                    
                    # Calculate weighted predicted volatility
                    predicted_volatility = sum(
                        weights[s] * stock_predictions[s]['annualized'] 
                        for s in selected_stocks 
                        if stock_predictions[s] is not None
                    )
                    
                    # Get risk status
                    risk_level, risk_color, risk_explanation = calculate_risk_status(
                        predicted_volatility,
                        max_drawdown
                    )
                    
                    # Create analysis tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Risk Analysis",
                        "📈 Portfolio Metrics",
                        "📉 Historical Performance",
                        "🔄 Portfolio Rebalancing"
                    ])
                    
                    with tab1:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.plotly_chart(
                                create_risk_gauge(
                                    volatility,
                                    predicted_volatility,
                                    risk_level,
                                    risk_color
                                ),
                                use_container_width=True
                            )
                        
                        with col2:
                            fig = px.pie(
                                values=list(investment_amounts.values()),
                                names=list(investment_amounts.keys()),
                                title=f"Total Investment: ${total_investment:,.2f}",
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"### Risk Status: <span style='color: {risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                        st.markdown(risk_explanation)
                        
                        st.subheader("Individual Stock Analysis")
                        stock_analysis = []
                        for stock in selected_stocks:
                            stock_data = selected_data[selected_data['Ticker'] == stock]
                            metrics_stock = calculate_stock_metrics(stock_data, model, scaler, features)
                            metrics_stock['Stock'] = stock
                            metrics_stock['Weight'] = f"{weights[stock]:.1%}"
                            stock_analysis.append(metrics_stock)
                        
                        st.dataframe(
                            pd.DataFrame(stock_analysis).set_index('Stock'),
                            use_container_width=True
                        )
                        
                        st.subheader("Stock Correlation Analysis")
                        st.plotly_chart(
                            create_correlation_heatmap(returns_data),
                            use_container_width=True
                        )
                    
                    with tab2:
                        st.subheader("Portfolio Metrics")
                        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        vol_explanation, dd_explanation = explain_metrics(
                            total_investment,
                            volatility,
                            max_drawdown,
                            predicted_volatility
                        )
                        st.markdown(vol_explanation)
                        st.markdown(dd_explanation)
                    
                    with tab3:
                        normalized_prices = selected_data.pivot(
                            index='Date',
                            columns='Ticker',
                            values='Close'
                        ).apply(lambda x: x / x.iloc[0])
                        
                        fig = px.line(
                            normalized_prices,
                            title="Normalized Price Performance (Starting Value = 1)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        rolling_vol = returns_data.rolling(window=30).std() * np.sqrt(252)
                        fig_vol = px.line(
                            rolling_vol,
                            title="30-Day Rolling Volatility (Annualized)"
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                        
                        portfolio_returns = returns_data.dot(pd.Series(weights))
                        drawdowns = (
                            (1 + portfolio_returns).cumprod()
                            .pipe(lambda x: (x - x.cummax()) / x.cummax())
                        )
                        
                        fig_dd = px.line(
                            drawdowns.reset_index(),
                            x='Date',
                            y=0,
                            title="Historical Drawdowns"
                        )
                        fig_dd.update_traces(fill='tonexty')
                        st.plotly_chart(fig_dd, use_container_width=True)
                    
                    with tab4:
                        st.subheader("Portfolio Rebalancing Analysis")
                        try:
                            optimal_weights = suggest_portfolio(
                                data,
                                returns_data,
                                volatility,
                                model,
                                scaler,
                                features
                            )
                            
                            weight_comparison = pd.DataFrame({
                                'Current Weight': weights,
                                'Suggested Weight': optimal_weights,
                                'Difference': {k: optimal_weights[k] - weights[k] for k in weights}
                            })
                            
                            st.dataframe(
                                weight_comparison.style.format("{:.2%}").background_gradient(
                                    subset=['Difference'],
                                    cmap='RdYlGn'
                                ),
                                use_container_width=True
                            )
                            
                            trades = {}
                            for stock in weights:
                                current_value = investment_amounts[stock]
                                target_value = optimal_weights[stock] * total_investment
                                trades[stock] = target_value - current_value
                            
                            st.subheader("Suggested Trades")
                            trades_df = pd.DataFrame({
                                'Stock': trades.keys(),
                                'Action': ['Buy' if t > 0 else 'Sell' for t in trades.values()],
                                'Amount': [abs(t) for t in trades.values()]
                            })
                            
                            st.dataframe(
                                trades_df.style.format({
                                    'Amount': '${:,.2f}'
                                }),
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Could not calculate optimal portfolio weights: {str(e)}")
    
    else:  # Suggested Portfolio
        st.sidebar.header("Risk Preference")
        
        target_risk_range = st.sidebar.slider(
            "Select Target Monthly Volatility Range:",
            min_value=0,
            max_value=50,
            value=(0,15),
            step=5,
            format="%d%%",
            help="Lower values indicate more conservative portfolios, higher values indicate more aggressive portfolios"
        )
        target_risk_min = target_risk_range[0] / 100
        target_risk_max = target_risk_range[1] / 100
        
        num_stocks = st.sidebar.number_input(
            "Number of stocks in portfolio:",
            min_value=2,
            max_value=10,
            value=5,
            help="More stocks generally provide better diversification"
        )
        
        if st.sidebar.button("🎯 Generate Suggested Portfolio"):
            with st.spinner("Generating ML-optimized portfolio..."):
                try:
                    selected_stocks = get_optimal_portfolio(
                        data,
                        target_risk_min,
                        target_risk_max,
                        num_stocks,
                        model,
                        scaler,
                        features
                    )
                    
                    if not selected_stocks:
                        st.error("Could not find suitable stocks matching your criteria")
                        return
                    
                    selected_data = data[data['Ticker'].isin(selected_stocks)].copy()
                    returns_data = selected_data.pivot(
                        index='Date',
                        columns='Ticker',
                        values='Close'
                    ).pct_change()
                    
                    target_risk_mid = (target_risk_min + target_risk_max) / 2
                    suggested_weights = suggest_portfolio(
                        data,
                        returns_data,
                        target_risk_mid,
                        model,
                        scaler,
                        features
                    )

                    suggested_weights = {k: float(v) for k, v in suggested_weights.items()}

                    st.subheader("🎯 ML-Optimized Portfolio Suggestion")
                    
                    tab1, tab2, tab3 = st.tabs([
                        "Portfolio Composition",
                        "Risk Analysis",
                        "Historical Performance"
                    ])
                    
                    with tab1:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            latest_prices = data[data['Ticker'].isin(selected_stocks)].groupby('Ticker')['Close'].last()
                            weights_df = pd.DataFrame({
                                'Stock': suggested_weights.keys(),
                                'Weight': [f"{w * 100:.1f}%" for w in suggested_weights.values()],
                                'Latest Price': [f"${latest_prices[s]:.2f}" for s in suggested_weights.keys()],
                                'Suggested Investment': [
                                    f"${w * 100000:,.2f}" for w in suggested_weights.values()
                                ]
                            })
                            st.dataframe(weights_df, use_container_width=True)
                        
                        with col2:
                            fig = px.pie(
                                values=list(suggested_weights.values()),
                                names=list(suggested_weights.keys()),
                                title="Portfolio Allocation",
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        stock_predictions = []
                        for stock in selected_stocks:
                            stock_data = data[data['Ticker'] == stock]
                            metrics = calculate_stock_metrics(stock_data, model, scaler, features)
                            metrics['Stock'] = stock
                            metrics['Portfolio Weight'] = f"{suggested_weights[stock]:.1%}"
                            stock_predictions.append(metrics)
                        
                        st.subheader("Individual Stock Risk Metrics")
                        metrics_df = pd.DataFrame(stock_predictions).set_index('Stock')
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        st.subheader("Portfolio Correlation Analysis")
                        st.plotly_chart(create_correlation_heatmap(returns_data), use_container_width=True)
                        
                        investment_amounts = {
                            stock: suggested_weights[stock] * 100000
                            for stock in selected_stocks
                        }
                        
                        predicted_volatilities = [
                            float(metrics['Predicted Volatility (Monthly)'].strip('%')) / 100
                            if metrics['Predicted Volatility (Monthly)'] != 'N/A'
                            else None
                            for metrics in stock_predictions
                        ]
                        
                        portfolio_metrics, portfolio_vol, max_dd = create_metrics_dashboard(
                            returns_data,
                            investment_amounts,
                            predicted_volatilities
                        )
                        
                        st.subheader("Portfolio-Level Risk Metrics")
                        metrics_display = pd.DataFrame(portfolio_metrics.items(), columns=['Metric', 'Value'])
                        st.dataframe(metrics_display, use_container_width=True)
                        
                    with tab3:
                        st.subheader("Historical Performance Analysis")
                        
                        # Normalized price performance
                        normalized_prices = selected_data.pivot(
                            index='Date',
                            columns='Ticker',
                            values='Close'
                        ).apply(lambda x: x / x.iloc[0])
                        
                        fig_perf = px.line(
                            normalized_prices,
                            title="Normalized Price Performance (Starting Value = 1)"
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                        
                        # Rolling volatility
                        rolling_vol = returns_data.rolling(window=30).std() * np.sqrt(252)
                        fig_vol = px.line(
                            rolling_vol,
                            title="30-Day Rolling Volatility (Annualized)"
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                        
                        # Portfolio drawdown analysis
                        portfolio_returns = returns_data.dot(pd.Series(suggested_weights))
                        drawdowns = (
                            (1 + portfolio_returns).cumprod()
                            .pipe(lambda x: (x - x.cummax()) / x.cummax())
                        )
                        
                        fig_dd = px.line(
                            drawdowns.reset_index(),
                            x='Date',
                            y=0,
                            title="Historical Drawdowns"
                        )
                        fig_dd.update_traces(fill='tonexty')
                        st.plotly_chart(fig_dd, use_container_width=True)
                        
                        # Risk metrics explanation
                        vol_explanation, dd_explanation = explain_metrics(
                            100000,  # Based on $100,000 portfolio
                            portfolio_vol,
                            max_dd,
                            float(portfolio_metrics.get('Predicted Volatility (Annualized)', '0%').strip('%'))/100
                        )
                        
                        st.markdown("### Risk Metrics Explanation")
                        st.markdown(vol_explanation)
                        st.markdown(dd_explanation)
                        
                except Exception as e:
                    st.error(f"Error in portfolio optimization: {str(e)}")

# ------------------ Main Application ------------------

def main():
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to", 
        [
            "Portfolio Risk Analyzer",
            "AI Portfolio Assistant",
            "Voice AI Assistant",
            "Portfolio Model Admin Dashboard"
        ],
        index=0
    )
    
    if page == "Portfolio Model Admin Dashboard":
        portfolio_model_admin_dashboard()
    elif page == "AI Portfolio Assistant":
        ai_portfolio_assistant()
    elif page == "Portfolio Risk Analyzer":
        portfolio_risk_analyzer()
    elif page == "Voice AI Assistant":
        voice_ai_assistant()

if __name__ == "__main__":
    main()
