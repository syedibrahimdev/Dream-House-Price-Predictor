import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import engineer_features

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'processed', 'train_cleaned.csv')
df = pd.read_csv(file_path)

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.6,
        colsample_bytree=0.6,
        n_jobs=-1,
        random_state=42
    ))
])

print("Training Final Model...")
model.fit(X, y)
joblib.dump(model, 'models/model.joblib')
print("Model saved using Modular Code!")