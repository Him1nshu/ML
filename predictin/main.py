# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)


# =========================
# 2. Load Data
# =========================
df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# =========================
# 3. Explore Data (glimpse equivalent)
# =========================
print("\n--- Train Info ---")
df_train.info()

print("\n--- Train Head ---")
print(df_train.head())

print("\n--- Missing Values ---")
print(df_train.isna().sum())


# =========================
# 4. (Optional) Date Feature Handling
# =========================
# Uncomment & modify if date column exists
# df_train["date"] = pd.to_datetime(df_train["date"])
# df_train["year"] = df_train["date"].dt.year
# df_train["month"] = df_train["date"].dt.month
# df_train["day"] = df_train["date"].dt.day
# df_train.drop(columns=["date"], inplace=True)


# =========================
# 5. Define Features & Target
# =========================
TARGET = "target"   # <-- CHANGE THIS to your actual target column

X = df_train.drop(columns=[TARGET])
y = df_train[TARGET]


# =========================
# 6. Train / Validation Split
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 7. Random Forest Model (ranger equivalent)
# =========================
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_valid)

print("\n--- Random Forest Performance ---")
print("RMSE:", mean_squared_error(y_valid, rf_preds, squared=False))
print("R2:", r2_score(y_valid, rf_preds))


# =========================
# 8. XGBoost Model
# =========================
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_valid)

print("\n--- XGBoost Performance ---")
print("RMSE:", mean_squared_error(y_valid, xgb_preds, squared=False))
print("R2:", r2_score(y_valid, xgb_preds))


# =========================
# 9. Train Final Model on Full Data
# =========================
xgb_model.fit(X, y)


# =========================
# 10. Predict on Test Data
# =========================
test_predictions = xgb_model.predict(df_test)

df_submission = pd.DataFrame({
    "prediction": test_predictions
})

df_submission.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")


# =========================
# 11. Visualization (ggplot equivalent)
# =========================
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=30, kde=True)
plt.title("Target Distribution")
plt.show()
