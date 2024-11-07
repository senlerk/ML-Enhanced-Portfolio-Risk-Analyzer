Here is the updated README file, incorporating the **Features** and **Technical Details** sections:

---

# ML-Enhanced Portfolio Risk Analyzer

The **ML-Enhanced Portfolio Risk Analyzer** is an advanced tool that integrates financial data, machine learning, and modern APIs to analyze portfolio risk, provide AI-powered investment insights, and optimize portfolio strategies. Designed with a user-friendly interface, it caters to both novice and professional investors.

---

## Table of Contents

1. [Features](#features)
2. [How to Use](#how-to-use)
   - [Clone the Repository](#1-clone-the-repository)
   - [Install Dependencies](#2-install-dependencies)
   - [Set Up Secrets and API Keys](#3-set-up-secrets-and-api-keys)
   - [Run the Application](#4-run-the-application)
   - [Explore the Features](#5-explore-the-features)
3. [Technical Details](#technical-details)
4. [Key Functionalities](#key-functionalities)
5. [File Structure](#file-structure)
6. [Docker Deployment](#docker-deployment)
7. [Notes](#notes)
8. [Technologies Used](#technologies-used)
9. [License](#license)

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

## How to Use

### 1. Clone the Repository

Clone the repository to your local environment:
```bash
git clone https://github.com/senlerk/ML-Enhanced-Portfolio-Risk-Analyzer
cd ML-Enhanced-Portfolio-Risk-Analyzer
```

---

### 2. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

### 3. Set Up Secrets and API Keys

To use the full functionality of the app, you need to configure a `secrets.toml` file for secure storage of sensitive data.

#### Steps to Configure:
1. Navigate to the `.streamlit` directory (create it if it doesn't exist).
2. Create a file named `secrets.toml`.
3. Add the following content:

```toml
[admin]
password = "your_admin_password"

[api_keys]
openai_api_key = "your_openai_api_key"
fred_api_key = "your_fred_api_key"
```

- **Admin Password**: Secures the Admin Dashboard.
- **OpenAI API Key**: Enables AI-powered assistant features.
- **FRED API Key**: Provides access to economic indicators like the Consumer Sentiment Index and VIX.

⚠️ **Important**: Add `.streamlit/secrets.toml` to `.gitignore` to prevent accidental uploads:
```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

---

### 4. Run the Application

Start the application locally:
```bash
streamlit run app.py
```

---

### 5. Explore the Features

Use the sidebar to navigate between sections:

- **Portfolio Risk Analyzer**: Assess portfolio risk, analyze performance, and visualize metrics such as historical volatility and maximum drawdowns.
- **AI Portfolio Assistant**: Engage with an OpenAI-powered assistant for personalized investment insights and financial strategies.
- **Voice AI Assistant**: Record voice queries and receive spoken responses for your financial questions.
- **Admin Dashboard**: Upload datasets, retrain machine learning models, and monitor system performance.

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

## Key Functionalities

### 1. Machine Learning for Risk Prediction
- Trains a **Random Forest Regressor** using over 20 financial and technical indicators.
- Integrates macroeconomic data (e.g., bond yields, sentiment indices) from the **FRED API**.
- Predicts future volatility and generates actionable insights for portfolio optimization.

### 2. Data Sources and APIs
- **OpenAI API**: Provides natural language understanding for the AI and voice assistants.
- **FRED API**: Supplies macroeconomic and financial risk metrics.
- **YFinance**: Fetches historical stock prices and data for S&P 500 companies.

### 3. Interactive Visualizations
- **Plotly Dashboards**: Visualize historical returns, volatility, and drawdowns.
- **Correlation Heatmaps**: Analyze relationships between portfolio assets.
- **Risk Gauges**: Intuitively assess risk levels and portfolio health.

---

## File Structure

```plaintext
.
├── app.py                      # Main Streamlit application
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
├── Dockerfile                  # Docker configuration for deployment
├── .streamlit/
│   └── secrets.toml            # Secure file for sensitive credentials
└── models/
    ├── random_forest_model.pkl # Trained machine learning model
    ├── scaler.pkl              # Feature scaler
    └── model_features.txt      # List of features used for model training
```

---

## Docker Deployment

Run the application in a Docker container:

1. Build the Docker image:
   ```bash
   docker build -t portfolio-risk-analyzer .
   ```

2. Start the Docker container:
   ```bash
   docker run -p 8501:8501 portfolio-risk-analyzer
   ```

---

## Notes

- **Security**: Always keep `secrets.toml` private and secure. Do not share API keys or sensitive credentials.
- **API Rate Limits**: Monitor usage of the OpenAI and FRED APIs to avoid exceeding rate limits.

---

## Technologies Used

- **Python Libraries**:
  - Data Handling: Pandas, Numpy, YFinance
  - Machine Learning: Scikit-learn, Joblib
  - Visualization: Plotly, Streamlit
- **APIs**:
  - OpenAI API for GPT-powered assistants.
  - FRED API for macroeconomic data.
- **Deployment**: Docker

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let me know if you'd like this saved as a downloadable file!
