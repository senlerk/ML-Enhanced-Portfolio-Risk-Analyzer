# Use Python 3.10 slim image instead of 3.9
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application files and model files
COPY v4.py .
COPY random_forest_model.pkl .
COPY scaler.pkl .
COPY model_features.txt .
COPY merged_financial_data.csv .
COPY fred_data.csv .
COPY sp500_data.csv .

# Copy Streamlit configuration and secrets file
COPY .streamlit/ .streamlit/

# Expose the port Streamlit will run on
EXPOSE 8080

# Set Streamlit configuration as environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application
CMD ["streamlit", "run", "v4.py"]
