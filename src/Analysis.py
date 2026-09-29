import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load Data
df = pd.read_csv('Data/Telco-Customer-Churn.csv')

# Basic Info
print(df.head(10))
df.info()
print(df.shape)

# Data Manipulation
male_customers = df[df["gender"] == "Male"]
print("Total Male Customers:", male_customers.shape[0])

dsl_customers = df[df["InternetService"] == "DSL"]
print("Total DSL Customers:", dsl_customers.shape[0])

new_customer = df[
    (df["gender"] == "Female") &
    (df["SeniorCitizen"] == 1) &
    (df["PaymentMethod"] == "Mailed check")
]
print("Total Records:", new_customer.shape[0])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

new_customer = df[
    (df["tenure"] < 10) |
    (df["TotalCharges"] < 500)
]
print("Total Records:", new_customer.shape[0])

# Visualization
churn_count = df["Churn"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(churn_count, labels=churn_count.index, autopct="%1.1f%%", startangle=90, explode=(0,0.1))
plt.title("Customer Churn Distribution")
plt.show()

internet_count = df["InternetService"].value_counts()

plt.figure(figsize=(7,5))
plt.bar(internet_count.index, internet_count.values)
plt.title("Distribution of Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.show()

# Preprocessing
df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

le = LabelEncoder()
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col].dtype):
        df[col] = le.fit_transform(df[col])

# =========================
# MODEL 1
# =========================
X = df[["tenure"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# CLASS WEIGHT (FIX)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

model = Sequential()
model.add(Dense(12, activation="relu", input_shape=(1,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_test, y_test),
    class_weight=class_weights,   # FIX
    verbose=1
)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.3).astype(int)   # FIX

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(cm).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Model 1 Accuracy vs Epochs")
plt.legend()
plt.show()

# =========================
# MODEL 2
# =========================
model2 = Sequential()
model2.add(Dense(12, activation='relu', input_shape=(1,)))
model2.add(Dropout(0.3))
model2.add(Dense(8, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history2 = model2.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_test, y_test),
    class_weight=class_weights,   # FIX
    verbose=1
)

y_pred2 = model2.predict(X_test)
y_pred2 = (y_pred2 > 0.3).astype(int)   # FIX

cm2 = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:\n", cm2)

ConfusionMatrixDisplay(cm2).plot()
plt.show()

accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy2)

plt.plot(history2.history["accuracy"], label="Train")
plt.plot(history2.history["val_accuracy"], label="Val")
plt.title("Model 2 Accuracy vs Epochs")
plt.legend()
plt.show()

# =========================
# MODEL 3 (WITH SCALING)
# =========================
from sklearn.preprocessing import StandardScaler

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

scaler = StandardScaler()
X = scaler.fit_transform(X)   # FIX

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

model3 = Sequential()
model3.add(Dense(12, activation='relu', input_shape=(3,)))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history3 = model3.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_test, y_test),
    class_weight=class_weights,   # FIX
    verbose=1
)

y_pred3 = model3.predict(X_test)
y_pred3 = (y_pred3 > 0.3).astype(int)   # FIX

cm3 = confusion_matrix(y_test, y_pred3)
print("Confusion Matrix:\n", cm3)

ConfusionMatrixDisplay(cm3).plot()
plt.show()

accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy:", accuracy3)

plt.plot(history3.history["accuracy"], label="Train")
plt.plot(history3.history["val_accuracy"], label="Val")
plt.title("Model 3 Accuracy vs Epochs")
plt.legend()
plt.show()

# Final Comparison
print("Model 1 Accuracy:", accuracy)
print("Model 2 Accuracy:", accuracy2)
print("Model 3 Accuracy:", accuracy3)