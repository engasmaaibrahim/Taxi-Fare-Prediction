import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
from joblib import dump

# Load Data
data = pd.read_csv('Fare Prediction.csv')

# Data Cleaning
data.columns = data.columns.str.strip()
data = data.drop(columns=['User Name', 'Driver Name', 'pickup_datetime', 'User ID', 'key',
                          'bearing', 'Weather', 'pickup_longitude', 'pickup_latitude',
                          'dropoff_longitude', 'dropoff_latitude'])
data = data.dropna()

# Encode Categorical Variables
car_condition_mapping = {"Excellent": 3, "Very Good": 2, "Good": 1, "Bad": 0}
traffic_condition_mapping = {"Congested Traffic": 2, "Dense Traffic": 1, "Flow Traffic": 0}

data["Car Condition"] = data["Car Condition"].map(car_condition_mapping)
data["Traffic Condition"] = data["Traffic Condition"].map(traffic_condition_mapping)

# Remove outliers
data = data[(data["fare_amount"] > 0) & (data["passenger_count"] > 0)]

# Feature Engineering
data["distance_traffic"] = data["distance"] * data["Traffic Condition"]
data["rush_hour"] = data["hour"].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
data["speed"] = data["distance"] / (data["hour"] + 1e-3)
data['trip_duration'] = data['distance'] / (data['speed'] + 1e-3)

# Handle Missing Values
data.fillna(data.median(), inplace=True)

from sklearn.decomposition import PCA

X = data.drop('fare_amount', axis=1)
y = data['fare_amount']

# Feature Scaling
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Get original feature names before PCA
original_feature_names = list(X.columns)

#handle outliers
pca = PCA(n_components=0.95)
# Apply PCA to scaled numeric features only
data_pca = pca.fit_transform(X[numeric_features])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter Tuning with Optuna
def lgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    }
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(lgb_objective, n_trials=10)
best_lgb_params = study.best_params

# Train the Best Model
lgb_model = lgb.LGBMRegressor(**best_lgb_params, random_state=42)
lgb_model.fit(X_train, y_train)

# Save Model and Preprocessing Objects Correctly
dump(lgb_model, 'model.joblib')
dump(scaler, 'scaler.joblib')
dump(original_feature_names, 'features.joblib')

# Evaluate the Model
y_pred = lgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")