
# **ML-Enhanced Portfolio Risk Analyzer**
### **End-to-End Data Science Project**

## **Table of Contents**
1. [Overview](#overview)
2. [Problem Statement and Business Case](#problem-statement-and-business-case)
3. [Features](#features)
4. [Technical Details](#technical-details)
5. [How to Use](#how-to-use)
6. [Setting Up Secrets and API Keys](#setting-up-secrets-and-api-keys)
7. [Data Sources and Integration](#data-sources-and-integration)
8. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
9. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
10. [Model Building and Evaluation](#model-building-and-evaluation)
11. [Advanced Features](#advanced-features)
12. [Assumptions and Limitations](#assumptions-and-limitations)
13. [Deployment](#deployment)
14. [Key Achievements](#key-achievements)
15. [Future Enhancements](#future-enhancements)

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

You're absolutely correct—those sections are indeed repetitive. To improve clarity and conciseness, we can consolidate the instructions about setting up secrets and API keys into a single section, while ensuring all the necessary details are covered.

Here’s a revised version of the **How to Use** section to eliminate redundancy:

---

### How to Use

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/senlerk/ML-Enhanced-Portfolio-Risk-Analyzer
   cd ML-Enhanced-Portfolio-Risk-Analyzer
   ```

2. **Install Dependencies**  
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Secrets and API Keys**  
   To enable the full functionality of this app, you need to create a `.streamlit/secrets.toml` file. This file will store sensitive credentials securely.  
   
   **Steps to Set Up**:  
   - Navigate to the `.streamlit` directory (create it if it doesn't exist).  
   - Create a new file named `secrets.toml`.  
   - Add the following content:
     ```toml
     [admin]
     password = "your_admin_password"

     [api_keys]
     openai_api_key = "your_openai_api_key"
     fred_api_key = "your_fred_api_key"
     ```
   - Replace `your_admin_password`, `your_openai_api_key`, and `your_fred_api_key` with your actual credentials.  

   **Explanation**:
   - `admin.password`: Used to secure the Admin Dashboard.
   - `api_keys.openai_api_key`: Required for the OpenAI-powered AI assistant.
   - `api_keys.fred_api_key`: Required to fetch economic indicators from the FRED API.  

   ⚠️ **Important**: Keep this file secure. Do not upload it to any public repository. Add it to your `.gitignore` file:
   ```bash
   echo ".streamlit/secrets.toml" >> .gitignore
   ```

4. **Run the Application**  
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. **Navigate the Application**  
   Use the sidebar to access different sections of the application:  
   - **Portfolio Risk Analyzer**: Assess portfolio risk and visualize key metrics.  
   - **AI Portfolio Assistant**: Chat with an AI-powered assistant for personalized investment advice.  
   - **Voice AI Assistant**: Use voice commands to interact with the app.  
   - **Admin Dashboard**: Manage datasets, retrain models, and perform administrative tasks.  

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

## **Deployment**
- **Local Deployment**:
   - Run using Streamlit on a local machine.
- **Cloud Deployment**:
   - Docker container is compatible with cloud platforms like Google Cloud.
   - Model retraining and dataset updates can be managed via the admin dashboard.

---

