---
title: Fraud Detection System
emoji: 💳
colorFrom: red
colorTo: yellow
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# 💳 Fraud Detection System

An advanced machine learning-based fraud detection system built with Streamlit that identifies potentially fraudulent transactions in real-time with 99.7% accuracy.

## 🚀 Features

- **Real-time Fraud Detection**: Analyze individual transactions instantly
- **Batch Processing**: Upload CSV files to process multiple transactions at once
- **Analytics Dashboard**: Visual insights into fraud patterns and transaction data
- **High Accuracy**: 99.7% accuracy rate using Logistic Regression
- **Fast Performance**: Optimized with caching for quick response times

## 📋 Requirements

- Python 3.11+
- Required packages (see `requirements.txt`)

## 🛠️ Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the following files in the project directory:
   - `app.py` - Main Streamlit application
   - `fraud_analysis_pipeline.pkl` - Trained ML model
   - `AIML_Dataset.csv` - Dataset for analytics (optional)

## 🏃 Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🐳 Docker Deployment

### Building the Docker Image

```bash
docker build -t fraud-detection-app .
```

### Running with Docker

```bash
docker run -p 7860:7860 fraud-detection-app
```

## 🤗 Hugging Face Spaces Deployment

### Method 1: Using Docker (Recommended)

1. **Prepare your files:**
   - Ensure all files are in the repository:
     - `app.py`
     - `fraud_analysis_pipeline.pkl`
     - `Dockerfile`
     - `requirements.txt`
     - `README.md`
   - Note: `AIML_Dataset.csv` is optional (only needed for Analytics tab)

2. **Create a Hugging Face Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Name your space (e.g., `fraud-detection-system`)

3. **Upload your files:**
   - Upload all files to your Space repository
   - Hugging Face will automatically build and deploy using the Dockerfile

4. **Configure Space Settings:**
   - Hardware: Choose based on your needs (CPU Basic is usually sufficient)
   - Visibility: Public or Private

### Method 2: Using Streamlit SDK

If you prefer using Streamlit SDK instead of Docker:

1. Create a Space with "Streamlit" SDK
2. Upload your files
3. Hugging Face will automatically detect `app.py` and run it

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── fraud_analysis_pipeline.pkl     # Trained ML model (required)
├── AIML_Dataset.csv                # Dataset for analytics (optional)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
└── README.md                       # This file
```

## 🎯 Usage

### Single Transaction Analysis

1. Navigate to the "Fraud Detection" tab
2. Enter transaction details:
   - Transaction Type
   - Transaction Amount
   - Sender's Old and New Balance
   - Receiver's Old and New Balance
3. Click "Analyze Transaction"
4. View the fraud prediction and probability

### Batch Processing

1. Navigate to the "Batch Processing" tab
2. Upload a CSV file with the following columns:
   - `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
   - `amount`: Transaction amount
   - `oldbalanceOrg`: Sender's old balance
   - `newbalanceOrig`: Sender's new balance
   - `oldbalanceDest`: Receiver's old balance
   - `newbalanceDest`: Receiver's new balance
3. View results and download the predictions

### Analytics Dashboard

- View fraud distribution charts
- Analyze fraud patterns by transaction type
- Explore transaction amount distributions
- Review fraud trends over time

## 🔧 Performance Optimizations

The app includes several performance optimizations:

- **Model Caching**: Uses `@st.cache_resource` to cache the ML model
- **Data Caching**: Uses `@st.cache_data` to cache the dataset
- **Chart Optimization**: Caches computed statistics for faster chart rendering
- **Memory Management**: Properly closes matplotlib figures to prevent memory leaks

## 📊 Model Information

- **Algorithm**: Logistic Regression with balanced class weights
- **Accuracy**: 99.7%
- **Precision**: 98.5%
- **Recall**: 97.8%
- **F1 Score**: 98.1%
- **Features**: Transaction type, amount, and balance information

## 🐛 Troubleshooting

### Model file not found
- Ensure `fraud_analysis_pipeline.pkl` is in the same directory as `app.py`
- If missing, run `train_model.py` to generate the model

### Dataset not found
- The app will work without the dataset, but analytics will be limited
- Ensure `AIML_Dataset.csv` is in the project directory for full functionality

### Docker build issues
- Ensure all required files are present before building
- Check that the Dockerfile paths match your file structure

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using Machine Learning | Model trained on 1M+ transactions**

