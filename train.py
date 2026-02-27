import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===== LOAD DATA =====
df = pd.read_csv("transactions.csv")

print("Dataset Columns:", df.columns)

# LAST COLUMN = TARGET
TARGET = df.columns[-1]
print("Target Column:", TARGET)

# ===== CLEAN DATA =====
df = df.fillna(0)

# ===== ENCODE TEXT =====
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ===== SPLIT =====
X = df.drop(TARGET, axis=1)
y = df[TARGET]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL =====
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# SAVE MODEL + FEATURES
pickle.dump((model, feature_names), open("model.pkl", "wb"))

print("Model saved successfully!")