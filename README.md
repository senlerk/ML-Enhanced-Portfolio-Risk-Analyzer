# **ML-Enhanced Portfolio Risk Analyzer**

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Usage Instructions](#usage-instructions)
5. [Important Notes](#important-notes)
6. [Technical Details](#technical-details)
7. [Future Enhancements](#future-enhancements)

---

## **Overview**
The **ML-Enhanced Portfolio Risk Analyzer** is a robust tool designed for portfolio risk assessment, optimization, and financial insights. It leverages machine learning models, advanced financial metrics, and interactive data visualizations to help users make informed investment decisions.

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

## **Setup and Installation**
### 1. Clone the Repository
```bash
git clone https://github.com/senlerk/ML-Enhanced-Portfolio-Risk-Analyzer
cd ML-Enhanced-Portfolio-Risk-Analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Secrets
Create a `.streamlit/secrets.toml` file in the project directory with the following content:
```toml
[admin]
password = "your_admin_password"

[api_keys]
openai_api_key = "your_openai_api_key"
fred_api_key = "your_fred_api_key"
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## **Usage Instructions**
### Initial Setup
1. **Error on First Run**:
   If you encounter the following error on the first run:
   ```
   Error loading ML components: [Errno 2] No such file or directory: 'random_forest_model.pkl'
   Failed to load ML components. Some features may be unavailable.
   ```
   This happens because the application doesn't have a pre-trained model loaded initially.

2. **Resolve the Error**:
   - Navigate to the **Admin Dashboard** page after starting the application.
   - Enter the Admin Password set in your `.streamlit/secrets.toml` file.
   - On the **Model Training** tab, run the steps to **gather data**, **merge data**, and **train the model**.
   - The application will save the required model files (`random_forest_model.pkl`, `scaler.pkl`, etc.).

3. **Restart Application**:
   - Restart the application after training the model to ensure all features are fully functional.

---

## **Important Notes**
- **Admin Access Required**:
   - The **Admin Dashboard** requires a password to ensure secure access to critical features like model training.
   - Use the password defined in the `.streamlit/secrets.toml` file.

- **API Keys**:
   - Ensure valid API keys for OpenAI and FRED are added to the `.streamlit/secrets.toml` file. Without them, certain functionalities will be unavailable.

---

## **Technical Details**
### **Tech Stack**
- **Frontend**:  
  Streamlit is used for creating an interactive and visually appealing user interface. The platform simplifies deployment and ensures cross-platform compatibility.
  
- **Backend**:  
  Python-based backend using:
  - **APIs**: Integration with OpenAI API (GPT-4) for AI-powered assistant functionalities and FRED API for economic indicators.
  - **Data Libraries**: 
    - `yfinance`: Fetch real-time stock data for S&P 500 companies.
    - `fredapi`: Access Federal Reserve data.
    - `BeautifulSoup`: Scrape additional stock information if needed.
    
- **Machine Learning**:  
  **Random Forest Regressor** model for predicting portfolio volatility. Includes advanced feature engineering with lagging, rolling averages, and interaction terms. ML pipeline uses `scikit-learn` for preprocessing, training, and model evaluation.

- **Visualization**:  
  Interactive charts powered by **Plotly**, including:
  - Risk Gauges
  - Correlation Heatmaps
  - Portfolio Composition Pie Charts
  - Time Series Analysis

- **Optimization**:  
  Portfolio optimization using `scipy.optimize.minimize` to calculate efficient allocations based on user-defined risk thresholds.

- **Data Sources**:
  - **S&P 500 Historical Stock Data**: Pulled from `yfinance`.
  - **FRED Economic Indicators**: Includes VIX, Consumer Sentiment Index, Yield Curve Spreads, and more.

- **Deployment**:  
  Dockerized application deployed on Google Cloud Run with the following configurations:
  - **Platform**: Managed
  - **Region**: `us-central1`
  - **Resources**: Increased RAM to 4GB for heavy computational tasks.
  - **Authentication**: Application allows public access with Google Cloud Run settings.

---

## **Future Enhancements**
1. **Integration of More ML Models**:
   - Explore additional algorithms like Gradient Boosting or Deep Learning for enhanced predictive accuracy.

2. **Advanced Portfolio Optimization**:
   - Implement methods like Monte Carlo Simulations or Black-Litterman for improved allocation strategies.

3. **Real-Time Data Integration**:
   - Fetch real-time financial data for instant analysis and insights.

4. **User Management System**:
   - Implement user accounts and role-based access for multi-user scenarios.

5. **Performance Enhancements**:
   - Utilize faster data handling libraries like Dask for large datasets.
