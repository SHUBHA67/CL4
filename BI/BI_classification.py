import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder

# ----------------------------------------
# Step 1: Load and Explore Data
# ----------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset Head:")
display(df.head())

print("\nDataset Info:")
print(df.info())

# Check class distribution
print("\nClass Distribution (Churn):")
print(df["Churn"].value_counts())

# ----------------------------------------
# Step 2: Data Preprocessing
# ----------------------------------------
# Drop irrelevant columns
df = df.drop("customerID", axis=1)

# Split features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"].map({'No': 0, 'Yes': 1})  # Encode target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X_train.select_dtypes(include=['object']).columns
oe = OrdinalEncoder()
X_train[categorical_cols] = oe.fit_transform(X_train[categorical_cols])

categorical_cols = X_test.select_dtypes(include=['object']).columns
X_test[categorical_cols] = oe.fit_transform(X_test[categorical_cols])
# ----------------------------------------
# Step 3: Train the Model
# ----------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------
# Step 4: Evaluate the Model
# ----------------------------------------
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
