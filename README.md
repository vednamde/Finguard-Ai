# FinGuard AI 💹

**Empowering Smarter Financial Decisions with Machine Learning**

FinGuard AI is an intelligent machine learning application that predicts the financial health of companies, helping stakeholders make informed decisions by identifying companies that are financially healthy or at risk of distress. Built with XGBoost and explainable AI techniques, it provides accurate predictions with transparent insights into the decision-making process.

## 🌟 Features

- **📊 Financial Health Prediction**: Classify companies as "Healthy" or "Distressed" based on key financial metrics
- **🎯 Risk Probability Scoring**: Get detailed risk probability percentages for each prediction
- **📈 Interactive Dashboard**: User-friendly Streamlit web interface with real-time visualizations
- **📄 Automated Reporting**: Generate comprehensive PDF and CSV reports with prediction results
- **🔍 Model Explainability**: SHAP-based feature importance analysis for transparent AI decisions
- **📂 CSV Upload Support**: Easy data ingestion through file upload functionality
- **📊 KPI Metrics**: Summary statistics including healthy vs. at-risk company counts and average health scores
- **📈 Health Score Visualization**: Bar charts showing financial health distribution across companies

## 🛠️ Tech Stack

- **Machine Learning**: XGBoost, scikit-learn, SHAP
- **Web Framework**: Streamlit
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, SHAP plots
- **Reporting**: FPDF (PDF generation)
- **Model Persistence**: joblib
- **Development**: Python 3.x

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/finhealth-ai.git
   cd finhealth-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

### For End Users

1. **Upload Data**: Click on the file uploader and select your CSV file containing company financial data
2. **View Predictions**: The app will automatically process the data and display predictions for each company
3. **Analyze Results**: Review the KPI metrics, prediction results, and health score visualizations
4. **Download Reports**: Generate and download PDF or CSV reports with your analysis results

### For Developers

#### Training the Model
```bash
python src/model_training.py
```

#### Generating SHAP Explanations
```bash
python src/explain_model.py
```

#### Data Processing
```bash
jupyter notebook notebooks/data-clean.ipynb
```

## 🧠 Model Details

### Algorithm
- **XGBoost Classifier**: Gradient boosting framework optimized for speed and performance
- **Hyperparameters**:
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 4
  - random_state: 42

### Performance Metrics
- **ROC-AUC Score**: [Insert current score from training]
- **Classification Report**: Includes precision, recall, and F1-score for both classes

### Feature Engineering
The model uses engineered financial ratios including:
- **Profitability Ratios**: Profit Margin, EBITDA Margin, ROA, ROE
- **Liquidity Ratios**: Cash Flow to Revenue, Liquidity Index
- **Efficiency Ratios**: Profit to Cash Flow, Revenue per Employee

## 📊 Data Pipeline

### 1. Data Cleaning (`notebooks/data-clean.ipynb`)
- Handle missing values (median imputation for market cap)
- Standardize column names
- Remove special characters from headers

### 2. Feature Engineering
- Create target variable: Financial_Status (1 = Distressed, 0 = Healthy)
- Calculate financial ratios from raw metrics
- Prepare features for machine learning

### 3. Model Training (`src/model_training.py`)
- Train-test split (80-20)
- XGBoost model training
- Model evaluation and saving

### 4. Explainability (`src/explain_model.py`)
- SHAP value calculation
- Feature importance visualization
- Summary plots saved to reports/

## 📁 Project Structure

```
finhealth-ai/
│
├── app/                          # Streamlit web application
│   ├── app.py                    # Main application file
│   ├── logo.png                  # Application logo
│   └── DejaVuSans.ttf           # Font file for PDF generation
│
├── src/                          # Source code
│   ├── model_training.py         # Model training script
│   └── explain_model.py          # SHAP explainability script
│
├── data/                         # Data files
│   ├── Financial Statements.csv  # Raw financial data
│   ├── Cleaned_Financial_Statements.csv    # Cleaned data
│   └── Engineered_Financial_Statements.csv # Feature engineered data
│
├── models/                       # Trained models
│   └── financial_health_xgb.pkl  # XGBoost model
│
├── reports/                      # Generated reports and visualizations
│   ├── shap_summary.png          # SHAP feature importance plot
│   └── prediction_results.csv    # Prediction outputs
│
├── notebooks/                    # Jupyter notebooks
│   └── data-clean.ipynb          # Data cleaning and feature engineering
│
├── assets/                       # Static assets
│   └── logo.png                  # Project logo
│
├── dejavu_fonts/                 # Font files for PDF generation
│
└── README.md                     # Project documentation
```

## 🤝 Contributing

We welcome contributions to FinGuard AI! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ved Namde**
- Built with ❤️ for smarter financial decision-making
- FinGuard AI © 2025

## 🙏 Acknowledgments

- XGBoost for the powerful gradient boosting framework
- SHAP for model interpretability
- Streamlit for the amazing web app framework
- The open-source community for continuous inspiration

---

**⭐ Star this repository if you find it helpful!**

For questions or support, please open an issue on GitHub.
