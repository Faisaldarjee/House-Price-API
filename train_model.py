
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load Kaggle training dataset
df = pd.read_csv("train.csv")

# Target column
if "SalePrice" not in df.columns:
    raise ValueError("SalePrice column not found in train.csv")

# Remove rows without target and separate features
df = df.dropna(subset=["SalePrice"])
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# Detect numeric and categorical columns
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess missing values and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Train regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)

# Default feature values for API requests with partial input
defaults = {}

for col in X.columns:
    if col in numeric_cols:
        value = X[col].median()
    else:
        modes = X[col].mode(dropna=True)
        value = modes.iloc[0] if not modes.empty else "Unknown"

    defaults[col] = value

joblib.dump({
    "model": model,
    "columns": X.columns.tolist(),
    "defaults": defaults
}, "model.joblib")

print("Training completed successfully.")
print(f"Training rows: {len(X)}")
print(f"Features: {len(X.columns)}")
print("Saved model.joblib with default feature values.")