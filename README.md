
# **ML-Enhanced Portfolio Risk Analyzer**
### **End-to-End Data Science Project**

## **Table of Contents**
1. [Overview](#overview)
2. [Problem Statement and Business Case](#problem-statement-and-business-case)
3. [Features](#features)
4. [Technical Details](#technical-details)
5. [How to Use](#how-to-use)
6. [Data Sources and Integration](#data-sources-and-integration)
7. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
9. [Model Building and Evaluation](#model-building-and-evaluation)
10. [Advanced Features](#advanced-features)
11. [Assumptions and Limitations](#assumptions-and-limitations)
12. [Deployment](#deployment)
13. [Key Achievements](#key-achievements)
14. [Future Enhancements](#future-enhancements)

---

## **Overview**
The **ML-Enhanced Portfolio Risk Analyzer** is a comprehensive end-to-end data science project designed to assist investors in analyzing portfolio risk, optimizing asset allocation, and predicting future volatility using advanced machine learning techniques. This project incorporates:
- **Historical financial data**
- **Machine learning-based predictions**
- **Risk visualization tools**
- **Generative AI-powered advisory systems**
- **Voice-activated AI assistant** 

It transforms complex financial insights into a user-friendly interactive application.

---

## **Problem Statement and Business Case**
Managing risk is critical for both novice and professional investors. A portfolio’s risk can vary significantly based on market conditions, volatility, and economic uncertainty. 

This application addresses:
1. **Portfolio risk assessment**: Evaluate current portfolio risk using historical and predicted metrics.
2. **Investment optimization**: Suggest optimized portfolios based on a target risk range.
3. **Financial education**: Provide actionable insights and education on portfolio management via AI.

**Target Audience**: Retail investors, financial advisors, and institutions seeking data-driven investment decisions.

---

## **Features**
### 1. **Portfolio Risk Analyzer**
   - Analyze risk for a custom-selected portfolio.
   - Metrics include annualized volatility, Sharpe ratio, maximum drawdown, and more.
   - Generates visualizations such as risk gauges, correlation heatmaps, and drawdown graphs.

### 2. **AI Portfolio Assistant**
   - Chat interface to answer portfolio and investment-related queries.
   - Powered by OpenAI GPT-4 for comprehensive financial advice.

### 3. **Voice AI Assistant**
   - Voice-activated AI for portfolio guidance.
   - Records, transcribes, and responds to user queries with audio playback.

### 4. **Portfolio Model Admin Dashboard**
   - Allows admins to update datasets, retrain models, and manage portfolio analysis pipelines.
   - Fetches and integrates stock data and economic indicators in real-time.

### 5. **Portfolio Optimization**
   - Suggests portfolios based on user-defined risk preferences.
   - Incorporates predicted volatility using advanced machine learning models.

---

## **Technical Details**
### Tech Stack:
- **Frontend**: Streamlit
- **Backend**: Python (Sklearn, RandomForest, OpenAI API, FRED API, YFinance)
- **Visualization**: Plotly
- **Data Sources**: S&P 500 data, FRED economic data
- **Machine Learning**: Random Forest Regressor
- **Deployment**: Cloud-based (Docker compatible)

---

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Navigate to different sections:
   - **Portfolio Risk Analyzer**: Assess portfolio risk and visualize metrics.
   - **AI Portfolio Assistant**: Chat with the AI for investment advice.
   - **Voice AI Assistant**: Interact using voice commands for financial queries.
   - **Admin Dashboard**: Manage datasets and retrain models.

---

## **Data Sources and Integration**
### Data Sources:
1. **Stock Data**: Historical S&P 500 data fetched using the YFinance API.
2. **Economic Indicators**: Data from the Federal Reserve Economic Data (FRED) API, including:
   - Consumer Sentiment Index
   - VIX (Volatility Index)
   - Treasury Yields

### Integration:
- Data from both sources are merged using common `Date` and `Ticker` fields.
- Additional features such as moving averages, rolling volatility, and lagged variables are calculated for model training.

---

## **Data Cleaning and Preparation**
### Steps Taken:
1. **Missing Data Handling**:
   - Forward and backward filling of missing values.
   - Removal of stocks with insufficient data points.
2. **Outlier Treatment**:
   - Winsorization for extreme values.
3. **Feature Engineering**:
   - Creation of technical indicators (moving averages, price ranges, etc.).
   - Calculation of rolling volatility and percentile values.

---

## **Exploratory Data Analysis (EDA)**
### Tools Used:
- **Tableau Dashboard**:
   - Insights into dataset characteristics and distributions.
   - Visual representation of time-series data for stock prices and economic indicators.
- **Python Visualizations**:
   - Heatmaps for correlation analysis.
   - Line charts for historical price performance.

---

## **Model Building and Evaluation**
### Models:
1. **Random Forest Regressor**: Used for predicting future volatility.
   - Features: Moving averages, volatility metrics, macroeconomic indicators.
2. **Other Considered Models**: Support Vector Machines (SVM), Gradient Boosting.

### Evaluation Metrics:
- Root Mean Square Error (RMSE)
- R² Score

### Feature Importance:
- Importance of features such as VIX, Treasury Yields, and historical volatility visualized to explain the model’s decision-making.

---

## **Advanced Features**
### 1. **Voice AI Integration**:
   - Uses **gTTS** for speech synthesis and **Whisper API** for transcription.
   - Provides an accessible way to interact with the application.

### 2. **Portfolio Optimization**:
   - Optimizes asset allocation using **Scipy’s optimization library**.
   - Balances predicted and historical volatility for diversification.

### 3. **Generative AI (ChatGPT)**:
   - Provides in-depth financial advice and explanations.

### 4. **Deployment**:
   - Fully containerized with **Docker**.
   - Ready for cloud deployment using Google Cloud or AWS.

---

## **Assumptions and Limitations**
### Assumptions:
- All economic indicators are relevant for stock market performance.
- Predicted volatility follows similar historical patterns.

### Limitations:
- Limited to available data; potential gaps in emerging markets.
- Predictions assume no abrupt changes in market conditions.

---

## **Deployment**
- **Local Deployment**:
   - Run using Streamlit on a local machine.
- **Cloud Deployment**:
   - Docker container is compatible with cloud platforms like Google Cloud.
   - Model retraining and dataset updates can be managed via the admin dashboard.

---

## **Key Achievements**
1. **Mandatory Deliverables**:
   - **Dataset Integration**: Combined S&P 500 data with FRED economic indicators.
   - **Comprehensive Documentation**: This README serves as a guide for the project.
   - **EDA**: Conducted and visualized using Python and Tableau.
   - **Model Building**: Random Forest and hyperparameter tuning completed.
   - **Final Model**: Justified with evaluation metrics and predictions.

2. **Nice-to-Have Additions**:
   - Voice AI Assistant
   - Deployment via Docker
   - Portfolio optimization using predicted metrics

3. **Advanced Data Processing**:
   - Sophisticated handling of missing data and outliers.
   - Feature engineering for enhanced model performance.

4. **Code Quality**:
   - Modular structure.
   - Well-commented codebase.

---

## **Future Enhancements**
1. Add sentiment analysis using real-time news or social media data.
2. Enhance portfolio optimization with deep learning models.
3. Automate model retraining on new data uploads.
4. Expand to global stocks and economic indicators.

---

