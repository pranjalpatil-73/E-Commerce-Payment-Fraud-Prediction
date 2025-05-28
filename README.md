# E-Commerce-Payment-Fraud-Prediction



## Overview

This project develops a machine learning model to predict fraudulent transactions in an e-commerce setting. By leveraging historical transaction data, the model identifies suspicious activities before they lead to financial losses, thereby enhancing the security and reliability of online payment systems.

## Business Problem Statement

E-commerce platforms face a major issue with payment fraud. Fraudulent transactions cause direct financial losses, chargeback fees, operational costs, and customer distrust. Since fraudulent transactions are rare, traditional rule-based systems often fail to detect them.

### Project Goals:

* **Minimize Financial Losses**: Identify and flag fraudulent transactions early.
* **Improve Operational Efficiency**: Reduce manual review workloads.
* **Enhance Customer Trust**: Offer a secure payment environment.
* **Leverage Data**: Use historical data to detect complex fraud patterns.

## Solution & Features

A complete machine learning pipeline for fraud prediction:

### Key Features:

* **Data Loading & Initial Exploration**
* **Exploratory Data Analysis (EDA)**
* **Advanced Preprocessing**:

  * Label Encoding of categorical columns.
  * Handling missing values.
  * Feature engineering (e.g., balance differences, ratio features).
  * Feature scaling using StandardScaler.
* **Feature Selection**: SelectKBest with f\_classif.
* **Class Imbalance Handling**: SMOTE for oversampling minority class.
* **Model Training & Evaluation**:

  * Random Forest Classifier
  * Gradient Boosting Classifier
  * Logistic Regression
* **Evaluation Metrics**:

  * `classification_report`, `confusion_matrix`, `roc_auc_score`, `accuracy_score`, `f1_score`
* **Model Persistence**: Save best model using joblib.
* **Real-time Prediction**: `predict_fraud()` function with high-risk threshold flag.

## Technologies Used

* **Language**: Python 3.x
* **Data Handling**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn`, `imbalanced-learn (SMOTE)`
* **Model Persistence**: `joblib`

## Getting Started

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash
git clone https://github.com/your-username/E-Commerce-Payment-Fraud-Prediction.git
cd E-Commerce-Payment-Fraud-Prediction

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### Dataset

Download `onlinefraud.csv` from the data source (e.g., Kaggle) and place it in the project directory.

### Running the Project

```bash
jupyter notebook
```

* Open `E-Commerce_Payment_Success_Prediction.ipynb`
* Run all cells from top to bottom.

## Usage

* After running the notebook, `fraud_detection_model.pkl` is created.
* Use `predict_fraud()` to classify new transactions as fraud or not.

Example:

```python
sample_transaction = {
    'step': 10,
    'type': 'TRANSFER',
    'amount': 1000000,
    'nameOrig': 'C123456789',
    'oldbalanceOrg': 1000000,
    'newbalanceOrig': 0,
    'nameDest': 'C987654321',
    'oldbalanceDest': 0,
    'newbalanceDest': 1000000,
    'isFlaggedFraud': 0
}
prediction_result = predict_fraud(sample_transaction)
```

Output:

* `prediction`: True/False
* `probability`: Probability of fraud
* `is_high_risk`: Based on threshold (e.g., 0.7)

## Testing

* Run all notebook cells.
* Validate outputs visually (EDA), and numerically (metrics).
* Modify `sample_transaction` to test multiple cases.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Name**: \[Your Name]
**GitHub**: \[Your GitHub Profile Link]
**Email**: \[Your Email Address]
**LinkedIn** (optional): \[Your LinkedIn Profile Link]
