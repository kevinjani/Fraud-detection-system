import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Data Loading
df = pd.read_csv("Fraud_detection_dataset.csv")

# Data Sampling to prevent Out-of-Memory Errors while preserving 'isFraud' distribution
sample_fraction = 0.1

df_fraud = df[df['isFraud'] == 1]
df_non_fraud = df[df['isFraud'] == 0]

df_fraud_sampled = df_fraud.sample(frac=sample_fraction, random_state=42)
df_non_fraud_sampled = df_non_fraud.sample(frac=sample_fraction, random_state=42)

df = pd.concat([df_fraud_sampled, df_non_fraud_sampled]).reset_index(drop=True)

print(f'\n--- Dataset sampled to {sample_fraction*100}% (new shape: {df.shape}) ---\n')

# Feature Engineering Helper Lists
customer_names_list = [
    "Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince", "Ethan Hunt",
    "Fiona Gallagher", "George Costanza", "Hannah Montana", "Ivy Lee", "Jack Sparrow",
    "Karen Miller", "Liam Gallagher", "Mia Wallace", "Noah Davis", "Olivia White"
]

locations_list = [
    "New York, US", "London, GB", "Tokyo, JP", "Paris, FR", "Berlin, DE",
    "Sydney, AU", "Toronto, CA", "Rome, IT", "Madrid, ES", "Beijing, CN",
    "Mumbai, IN", "Cairo, EG", "Rio de Janeiro, BR", "Mexico City, MX", "Dubai, AE"
]

# Feature Engineering Function
def engineer_categorical_features(df_input):
    df_output = df_input.copy()

    # Generate random customer names for all (general variety, not pattern-based for fraud detection)
    df_output['generated_customer_name'] = np.random.choice(customer_names_list, size=len(df_output))

    # Define specific Fraudulent and Non-Fraudulent Merchant-Expense Patterns
    fraud_merchant_expense_patterns = [
        ("Digital Deception Ltd.", "Gambling"),
        ("Secret Money Makers", "Investments"),
        ("Instant Wealth Inc.", "Financial Services"),
        ("Guaranteed Profits Ltd.", "Subscription Fees")
    ]
    non_fraud_merchant_expense_patterns = [
        ("Global Solutions Inc.", "Groceries"),
        ("Tech Innovations Ltd.", "Electronics"),
        ("Creative Designs Co.", "Dining Out"),
        ("Rapid Delivery Corp.", "Clothing"),
        ("Elite Consulting Group", "Fuel"),
        ("EasyCash Loans", "Healthcare"),
        ("FastFunds Now", "Entertainment"),
        ("Pinnacle Ventures", "Online Shopping"),
        ("Synergy Global", "Travel"),
        ("Worldwide Holdings", "Utilities"),
        ("Zenith Corporation", "Education")
    ]

    # Identify fraud and non-fraud rows
    fraud_mask = df_output['isFraud'] == 1
    non_fraud_mask = df_output['isFraud'] == 0

    # Assign features for fraudulent transactions
    num_fraud = fraud_mask.sum()
    if num_fraud > 0:
        # Assign merchant and expense from fraud patterns
        chosen_fraud_patterns_indices = np.random.choice(len(fraud_merchant_expense_patterns), size=num_fraud)
        df_output.loc[fraud_mask, 'generated_merchant_name'] = [fraud_merchant_expense_patterns[i][0] for i in chosen_fraud_patterns_indices]
        df_output.loc[fraud_mask, 'expense_category'] = [fraud_merchant_expense_patterns[i][1] for i in chosen_fraud_patterns_indices]

        # Assign sender and receiver locations (guaranteed mismatch for fraud)
        sender_loc_for_fraud = np.random.choice(locations_list, size=num_fraud)
        receiver_loc_for_fraud = np.array([
            np.random.choice([loc for loc in locations_list if loc != s])
            if len([loc for loc in locations_list if loc != s]) > 0
            else s for s in sender_loc_for_fraud
        ])
        df_output.loc[fraud_mask, 'sender_location'] = sender_loc_for_fraud
        df_output.loc[fraud_mask, 'receiver_location'] = receiver_loc_for_fraud

    # Assign features for non-fraudulent transactions
    num_non_fraud = non_fraud_mask.sum()
    if num_non_fraud > 0:
        # Assign merchant and expense from non-fraud patterns
        chosen_non_fraud_patterns_indices = np.random.choice(len(non_fraud_merchant_expense_patterns), size=num_non_fraud)
        df_output.loc[non_fraud_mask, 'generated_merchant_name'] = [non_fraud_merchant_expense_patterns[i][0] for i in chosen_non_fraud_patterns_indices]
        df_output.loc[non_fraud_mask, 'expense_category'] = [non_fraud_merchant_expense_patterns[i][1] for i in chosen_non_fraud_patterns_indices]

        # Assign matching sender and receiver locations
        locations_for_non_fraud = np.random.choice(locations_list, size=num_non_fraud)
        df_output.loc[non_fraud_mask, 'sender_location'] = locations_for_non_fraud
        df_output.loc[non_fraud_mask, 'receiver_location'] = locations_for_non_fraud

    return df_output

# Apply the feature engineering function
df = engineer_categorical_features(df)

print("\n--- DataFrame with new generated feature columns (sample of fraud and non-fraud) ---\n")
# Display a few fraud and a few non-fraud examples
display(pd.concat([
    df[df['isFraud'] == 1].head(5),
    df[df['isFraud'] == 0].head(5)
])[['generated_customer_name', 'generated_merchant_name', 'expense_category', 'sender_location', 'receiver_location', 'isFraud']])

# Data Preprocessing and Feature Engineering for Logistic Regression

drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
df = df.drop(columns=drop_cols)

y = df['isFraud']
X = df.drop(columns=['isFraud'])

numerical_features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest'
]

categorical_features = [
    'type',
    'generated_customer_name', 'generated_merchant_name',
    'expense_category', 'sender_location', 'receiver_location'
]

X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\n--- Preprocessed Features (first 5 rows) ---\n")
display(X.head())

display(y.head())

# Model Training and Evaluation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Model Trained Successfully! ---")

# Evaluation Metrics
print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---\n")
fig1, ax1 = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n--- ROC Curve ---\n")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
ax2.plot([0, 1], [0, 1], linestyle='--', color='navy')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic (ROC) Curve")
ax2.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(f"\n**AUC Score: {roc_auc:.2f}**")
if roc_auc > 0.8:
    print("Excellent model!")
elif roc_auc > 0.7:
    print("Good model, consider further tuning.")
else:
    print("Model performance could be improved.")
