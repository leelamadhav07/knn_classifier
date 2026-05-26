import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# LOAD DATASET

df = pd.read_csv("data/heart.csv")

# FEATURES & TARGET

X = df[["age", "sex", "cp", "trestbps", "chol", "thalach"]]

y = df["target"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# MODEL TRAINING

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# ACCURACY

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# SAVE MODEL

with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# SAVE SCALER

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
