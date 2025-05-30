import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Suppress warnings from SHAP
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

# EDA - Descriptive Stats and Visualization
def perform_eda(df):
    print("Descriptive statistics:\n", df.describe())

    # Churn count
    churn_counts = df['Churn'].value_counts()
    churn_counts.plot(kind='barh', figsize=(8, 6))
    plt.xlabel("Count", labelpad=14)
    plt.ylabel("Target Variable", labelpad=14)
    plt.title("Count of TARGET Variable per category", y=1.02)
    plt.tight_layout()
    plt.show()

    # Percentage
    print("Churn distribution (%):\n", 100 * churn_counts / len(df['Churn']))
    print("Normalized Churn distribution:\n", df['Churn'].value_counts(normalize=True))

    # Missing values
    missing = pd.DataFrame((df.isnull().sum()) * 100 / df.shape[0]).reset_index()
    missing.columns = ['column_name', 'missing_percentage']
    plt.figure(figsize=(16, 5))
    sns.pointplot(x='column_name', y='missing_percentage', data=missing)
    plt.xticks(rotation=90, fontsize=7)
    plt.title("Percentage of Missing values")
    plt.ylabel("PERCENTAGE")
    plt.tight_layout()
    plt.show()

# SHAP analysis using XGBoost
def shap_analysis(df):
    # Encode categorical features
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    # Drop customerID if it exists
    if 'customerID' in df_encoded.columns:
        df_encoded.drop('customerID', axis=1, inplace=True)

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    print("Showing SHAP summary plot...")
    shap.summary_plot(shap_values, X_test)

# Main script
if __name__ == "__main__":
    filepath = r'C:\Users\Pooja\churn_pred\Churn_prediction\data\WA_Fn-UseC_-Telco-Customer-Churn (2).csv'
    df = load_data(filepath)

    perform_eda(df)

    print("\nRunning SHAP analysis...")
    shap_analysis(df)
