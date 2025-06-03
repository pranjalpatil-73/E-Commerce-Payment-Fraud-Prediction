# E-Commerce-Payment-Fraud-Prediction
## E-Commerce Payment Success Prediction

This project develops a machine learning model to predict the success or failure of e-commerce payment transactions. By analyzing various transaction attributes, user behavior, and payment gateway data, the system aims to identify factors contributing to payment failures, enabling businesses to optimize their payment processes, reduce friction, and enhance the overall customer experience.

---

### Importance of Solving This Problem

Predicting payment success in e-commerce is a critical capability with significant implications for business revenue and customer satisfaction:

* **Reduced Cart Abandonment:** Payment failures are a major cause of shopping cart abandonment. By predicting and potentially preventing these failures, businesses can significantly reduce lost sales.
* **Increased Conversion Rates:** A smooth and successful payment process is crucial for converting interested customers into paying customers. Optimizing this step directly boosts conversion rates.
* **Enhanced Customer Experience:** Frustrating payment experiences can lead to customer dissatisfaction and a reluctance to return. Predicting and mitigating failures ensures a seamless checkout, improving customer loyalty and brand perception.
* **Optimized Payment Gateway Routing:** Insights from the model can help direct transactions through payment gateways with higher success rates for specific transaction types, amounts, or customer profiles.
* **Revenue Protection:** Preventing payment failures directly protects potential revenue that would otherwise be lost.
* **Operational Efficiency:** Automated prediction helps prioritize failed transactions for manual review, if necessary, or informs automated retry mechanisms, reducing the burden on customer support.
* **Identification of Fraudulent or Risky Transactions:** While not solely a fraud detection project, patterns leading to payment failures can sometimes overlap with or indicate risky transactions, providing an additional layer of insight.

---

### Features

* **Data Loading and Initial Exploration:** Imports and provides an initial overview of the dataset.
* **Comprehensive Data Preprocessing:**

  * Handles numerical (e.g., `TransactionAmount`, `UserActivityScore`) and categorical features (e.g., `PaymentMethod`, `DeviceType`, `BillingCountry`).
  * Applies OneHotEncoder for categorical variables.
  * Uses StandardScaler for numerical feature scaling.
  * Manages missing values to maintain data quality.
* **Exploratory Data Analysis (EDA):** Visualizations (count plots, bar plots, heatmaps) to uncover relationships and key drivers of payment success or failure.
* **Machine Learning Model Training:**

  * Trains models such as `LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier` (and optionally `XGBoost`).
  * Uses Pipelines for streamlined workflows.
* **Hyperparameter Tuning:** Uses `GridSearchCV` for optimal hyperparameter selection.
* **Thorough Model Evaluation:**

  * Metrics: `classification_report`, `confusion_matrix`, and `ROC AUC score`.
  * Identifies the best-performing model via cross-validation.
* **Model Persistence:** Saves the trained model and preprocessing pipeline using `pickle`.
* **Payment Success Prediction Function:** A function (`predict_payment_success`) to predict the success status and probability for new transactions.

---

### Technologies Used

* **Python**: Core programming language.
* **Pandas, NumPy**: Data manipulation and numerical operations.
* **Matplotlib, Seaborn**: Data visualization.
* **Scikit-learn**: Machine learning library.

  * `train_test_split`, `GridSearchCV`.
  * `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`.
  * `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`.
  * `classification_report`, `confusion_matrix`, `roc_auc_score`.
* **XGBoost (optional)**: For advanced boosting models.
* **Pickle**: For model serialization.

---

### Getting Started

#### Prerequisites

* Python 3.7+
* Jupyter Notebook or JupyterLab

#### Installation

1. Download the Notebook and Dataset:

   * `E-Commerce_Payment_Success_Prediction.ipynb`
   * Dataset file: `ecommerce_payments.csv`

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

*Note: `xgboost` is optional.*

---

### Usage

1. Open Jupyter Notebook:

```bash
jupyter notebook "E-Commerce_Payment_Success_Prediction.ipynb"
```

2. Run all cells sequentially.

   * Loads and preprocesses data.
   * Performs EDA.
   * Trains and tunes models.
   * Saves the best model and preprocessor.
   * Defines `predict_payment_success` function.

3. Make predictions:

```python
new_transaction_data = {
    'TransactionID': 'T987654321',
    'TransactionAmount': 150.75,
    'PaymentMethod': 'Credit Card',
    'DeviceType': 'Mobile',
    'UserActivityScore': 0.85,
    'PreviousAttempts': 0,
    'IsFirstPurchase': 1,
    'BillingCountry': 'USA',
    'ShippingCountry': 'USA',
    'TimeOfDay': 'Evening',
    'UserAgent': 'Chrome'
}

# prediction_result = predict_payment_success(new_transaction_data, loaded_model, loaded_preprocessor)
# print("Payment Success Prediction:", prediction_result)
```

---

### Future Enhancements

* Real-time API integration.
* Anomaly detection for failure types.
* Dynamic payment routing.
* Customer-specific retries.
* External data sources (fraud scores, latency, gateway status).
* Interactive dashboard for monitoring and analysis.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Name**: \[Your Name]
**GitHub**: \[Your GitHub Profile Link]
**Email**: \[Your Email Address]
**LinkedIn** (optional): \[Your LinkedIn Profile Link]
