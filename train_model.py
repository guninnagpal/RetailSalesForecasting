# ======================================
# UNIVERSAL SCRIPT: CMD + JENKINS COMPATIBLE
# ======================================

import os
import pandas as pd
import numpy as np
import matplotlib

# Detect Jenkins environment (headless)
if "JENKINS_HOME" in os.environ:
    matplotlib.use('Agg')  # No GUI for Jenkins

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


# ======================================
# STEP 1: ENVIRONMENT DETECTION
# ======================================
RUNNING_ON_JENKINS = "JENKINS_HOME" in os.environ

print("===================================")
print("Environment Detection:")
if RUNNING_ON_JENKINS:
    print("Running on: JENKINS (Headless Mode)")
else:
    print("Running on: LOCAL CMD (GUI Mode)")
print("===================================")


# ======================================
# STEP 2: LOAD DATASET (Works Locally + Jenkins)
# ======================================
DATA_PATH = "housing.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "housing.csv not found. Place it in the same folder as train_model.py"
    )

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))


# ======================================
# STEP 3: HANDLE MISSING VALUES
# ======================================
print("\nChecking Missing Values:")
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))


# ======================================
# STEP 4: FEATURE ENGINEERING (High Accuracy)
# ======================================
print("\nPerforming Feature Engineering...")

df["rooms_per_household"] = df["total_rooms"] / (df["households"] + 1)
df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + 1)
df["population_per_household"] = df["population"] / (df["households"] + 1)
df["income_per_household"] = df["median_income"] / (df["households"] + 1)


# ======================================
# STEP 5: ENCODE CATEGORICAL COLUMN
# ======================================
if "ocean_proximity" in df.columns:
    print("Encoding categorical column: ocean_proximity")
    df = pd.get_dummies(df, drop_first=True)


# ======================================
# STEP 6: DEFINE FEATURES & TARGET
# ======================================
TARGET = "median_house_value"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Log transform for higher accuracy
y_log = np.log1p(y)


# ======================================
# STEP 7: TRAIN-TEST SPLIT
# ======================================
print("\nSplitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)


# ======================================
# STEP 8: MODEL (Optimized for Accuracy + Speed)
# ======================================
print("\nTraining XGBoost Model...")

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ======================================
# STEP 9: PREDICTIONS
# ======================================
print("\nMaking Predictions...")
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_val)


# ======================================
# STEP 10: ACCURACY METRICS
# ======================================
r2 = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
cv_scores = cross_val_score(model, X, y_log, cv=3, scoring='r2')

print("\n=========== MODEL PERFORMANCE ===========")
print("Validation Accuracy (R2):", round(r2 * 100, 2), "%")
print("Cross-Validation Accuracy:", round(cv_scores.mean() * 100, 2), "%")
print("RMSE:", rmse)
print("========================================")


# ======================================
# STEP 11: UNIVERSAL PLOTTING (CMD + Jenkins)
# ======================================
def save_or_show_plot(filename):
    """
    If Jenkins -> Save plots
    If Local CMD -> Show plots + Save
    """
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")

    if not RUNNING_ON_JENKINS:
        plt.show()  # Show only on local machine

    plt.close()


# Plot 1: Actual vs Predicted
plt.figure()
plt.scatter(y_actual, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
save_or_show_plot("plot1_actual_vs_predicted.png")


# Plot 2: Distribution
plt.figure()
plt.hist(y_actual, bins=40)
plt.xlabel("House Prices")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
save_or_show_plot("plot2_price_distribution.png")


# Plot 3: Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)

plt.figure()
top_features.plot(kind="bar")
plt.title("Top 10 Important Features")
save_or_show_plot("plot3_feature_importance.png")


print("\nExecution Completed Successfully!")
if RUNNING_ON_JENKINS:
    print("Plots saved in Jenkins Workspace.")
else:
    print("Plots displayed locally and saved in project folder.")