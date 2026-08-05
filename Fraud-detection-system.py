import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/Fraud_detection_dataset.csv")

sample_fraction = 0.1

df_fraud = df[df['isFraud'] == 1]
df_non_fraud = df[df['isFraud'] == 0]

df_fraud_sampled = df_fraud.sample(frac=sample_fraction, random_state=42)
df_non_fraud_sampled = df_non_fraud.sample(frac=sample_fraction, random_state=42)

df = pd.concat([df_fraud_sampled, df_non_fraud_sampled]).reset_index(drop=True)

print(f'Dataset sampled to {sample_fraction*100}% (new shape: {df.shape})\n')

customer_names_list = [
    "Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince", "Ethan Hunt",
    "Fiona Gallagher", "George Costanza", "Hannah Montana", "Ivy Lee", "Jack Sparrow",
    "Karen Miller", "Liam Gallagher", "Mia Wallace", "Noah Davis", "Olivia White"
]

merchant_names_for_generation = [
    "Digital Deception Ltd.", "Secret Money Makers", "Instant Wealth Inc.", "Guaranteed Profits Ltd.",
    "Global Solutions Inc.", "Tech Innovations Ltd.", "Creative Designs Co.", "Rapid Delivery Corp.",
     "Elite Consulting Group", "EasyCash Loans", "FastFunds Now", "Pinnacle Ventures",
     "Synergy Global", "Worldwide Holdings", "Zenith Corporation"
]

receiver_location = [
    "New York, US", "London, GB", "Tokyo, JP", "Paris, FR", "Berlin, DE",
    "Sydney, AU", "Toronto, CA", "Rome, IT", "Madrid, ES", "Beijing, CN",
    "Mumbai, IN", "Cairo, EG", "Rio de Janeiro, BR", "Mexico City, MX", "Dubai, AE"
]

# Initialize new columns with NaN and explicit object dtype
df['customer_name'] = pd.Series(np.nan, index=df.index, dtype='object')
df['merchant_name'] = pd.Series(np.nan, index=df.index, dtype='object')
df['receiver_location'] = pd.Series(np.nan, index=df.index, dtype='object')

# Populate the first 15 entries of the new columns
df.loc[df.index[:15], 'customer_name'] = customer_names_list[:15]
df.loc[df.index[:15], 'merchant_name'] = merchant_names_for_generation[:15]
df.loc[df.index[:15], 'receiver_location'] = receiver_location[:15]

# Drop the original nameOrig, nameDest, and isFlaggedFraud columns
df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

# Rename columns as requested
df = df.rename(columns={
    'type': 'transaction_type',
    'oldbalanceOrg': 'customer_old_balance',
    'newbalanceOrig': 'customer_new_balance',
    'oldbalanceDest': 'receiver_old_balance',
    'newbalanceDest': 'receiver_new_balance'
})

# Define the new desired column order to place the new columns
new_column_order = [
    'transaction_type', 'amount', 'customer_name','customer_old_balance', 'customer_new_balance',
    'merchant_name', 'receiver_old_balance', 'receiver_new_balance', 'receiver_location', 'isFraud'
]

# Reindex the DataFrame to set the new column order
df = df[new_column_order]

print("DataFrame after adding and reordering new columns (first 20 rows)\n")
display(df.head(20))

y = df['isFraud']
X = df.drop(columns=['isFraud', 'customer_name', 'merchant_name', 'receiver_location'])

numerical_features = [
    'amount', 'customer_old_balance', 'customer_new_balance',
    'receiver_old_balance', 'receiver_new_balance'
]

categorical_features = [
    'transaction_type'
]

X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("Preprocessed Features (first 5 rows)\n")
display(X.head())

display(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Logistic Regression Model Trained Successfully!\n")

print("Classification Report\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix\n")
fig1, ax1 = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("ROC Curve\n")
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

print(f"AUC Score: {roc_auc:.2f}\n")
if roc_auc > 0.8:
    print("Excellent model!")
elif roc_auc > 0.7:
    print("Good model, consider further tuning.")
else:
    print("Model performance could be improved.")
