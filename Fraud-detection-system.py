import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- Initial Setup, Data Loading, and EDA ----
# Load the dataset
df = pd.read_csv("Fraud_detection_dataset.csv")

print('--- Initial DataFrame Info ---\n')
df.info()

print('\n--- Initial Descriptive Statistics ---\n')
display(df.describe())

# --- Data Sampling to prevent Out-of-Memory Errors ---
# The dataset is very large (6.3 million rows), and one-hot encoding several categorical features
# can quickly lead to an 'Out of Memory' error in Colab's environment. To mitigate this,
# we will sample a subset of the data for processing and model training.
sample_fraction = 0.1 # Sample 10% of the data
df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
print(f'\n--- Dataset sampled to {sample_fraction*100}% (new shape: {df.shape}) ---\n')

# --- Feature Engineering: Adding New Categorical Columns ---

# Pre-generated list of 15 sample customer names (reduced for memory efficiency)
customer_names_list = [
    "Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince", "Ethan Hunt",
    "Fiona Gallagher", "George Costanza", "Hannah Montana", "Ivy Lee", "Jack Sparrow",
    "Karen Miller", "Liam Gallagher", "Mia Wallace", "Noah Davis", "Olivia White"
]

# Pre-generated list of 15 merchant names (mix of genuine and fraud merchants, reduced for memory efficiency)
merchant_names_list = [
    "Global Solutions Inc.", "Tech Innovations Ltd.", "Creative Designs Co.", "Rapid Delivery Corp.",
    "Elite Consulting Group", "EasyCash Loans", "FastFunds Now", "Instant Wealth Inc.",
    "Guaranteed Profits Ltd.", "Secret Money Makers", "Pinnacle Ventures", "Synergy Global",
    "Worldwide Holdings", "Zenith Corporation", "Digital Deception Ltd."
]
random.shuffle(merchant_names_list)

# Pre-generated list of 15 expense categories (reduced for memory efficiency)
expense_categories_list = [
    "Groceries", "Electronics", "Dining Out", "Clothing", "Fuel",
    "Healthcare", "Entertainment", "Online Shopping", "Financial Services", "Gambling",
    "Investments", "Subscription Fees", "Travel", "Utilities", "Education"
]
random.shuffle(expense_categories_list)

# Pre-generated list of 15 locations (cities and countries, reduced for memory efficiency)
locations_list = [
    "New York, US", "London, GB", "Tokyo, JP", "Paris, FR", "Berlin, DE",
    "Sydney, AU", "Toronto, CA", "Rome, IT", "Madrid, ES", "Beijing, CN",
    "Mumbai, IN", "Cairo, EG", "Rio de Janeiro, BR", "Mexico City, MX", "Dubai, AE"
]
random.shuffle(locations_list)

# Vectorized assignment of generated categorical features
df['generated_customer_name'] = np.random.choice(customer_names_list, size=len(df))
df['generated_merchant_name'] = np.random.choice(merchant_names_list, size=len(df))
df['expense_category'] = np.random.choice(expense_categories_list, size=len(df))
df['sender_location'] = np.random.choice(locations_list, size=len(df))

# Vectorized assignment for receiver_location with 20% chance of being different
df['receiver_location'] = df['sender_location'] # Default to matching sender_location (80% case)

# Create a mask for the 20% of rows where receiver_location should be different
mask_different_location = np.random.rand(len(df)) < 0.2

# For the rows in the mask, assign a different location
df.loc[mask_different_location, 'receiver_location'] = np.random.choice(locations_list, size=mask_different_location.sum())

print("\n--- DataFrame with new generated feature columns (first 10 rows) ---\n")
display(df[['generated_customer_name', 'generated_merchant_name', 'expense_category', 'sender_location', 'receiver_location', 'isFraud']].head(10))

# --- Data Preprocessing and Feature Engineering for Logistic Regression ---

# Define columns to drop as they are not needed for prediction or are the target
drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']  # 'nameOrig' and 'nameDest' are identifiers, 'isFlaggedFraud' is too close to target
df = df.drop(columns=drop_cols)

# Define target variable
y = df['isFraud']
# Define features (X) by dropping the target variable
X = df.drop(columns=['isFraud'])

# Define numerical and categorical features for separate processing
numerical_features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest'
]

categorical_features = [
    'type',
    'generated_customer_name', 'generated_merchant_name',
    'expense_category', 'sender_location', 'receiver_location'
]

# Apply One-Hot Encoding to categorical features using pd.get_dummies
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Apply StandardScaler to numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\n--- Preprocessed Features (X) Shape ---\n")
print(X.shape)

print("\n--- Preprocessed Features (first 5 rows) ---\n")
display(X.head())

print("\n--- Target Variable (y) Shape ---\n")
print(y.shape)

print("\n--- Target Variable (first 5 values) ---\n")
display(y.head())

# --- Model Training and Evaluation ---

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify to maintain fraud ratio

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42) # Using liblinear for smaller datasets or L1/L2 regularization
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Model Trained Successfully! ---\n")

# --- Evaluation Metrics ---
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
