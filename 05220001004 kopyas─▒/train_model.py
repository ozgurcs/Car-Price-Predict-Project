"""Train a vehicle price prediction model and build a similarity index.

Usage:
    python train_model.py --csv data/vehicle_prices.csv
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import joblib
from inspect import signature

from sklearn.model_selection import KFold

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise SystemExit(
        "LightGBM is required. Run `pip install lightgbm` and try again."
    )

# ----------------------------- #
# 1. Basic preprocessing helper #
# ----------------------------- #
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning – extend this as needed."""
    # Drop rows without a target
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    df = df.dropna(subset=['Price']).copy()

    # Example: extract numeric engine size "3.5L" -> 3.5
    if 'Engine' in df.columns:
        df['Engine'] = (
            df['Engine'].astype(str).str.extract(r'([0-9.]+)').astype(float)
        )
    # Example: convert odometer "145,000" -> 145000
    if 'Odometer' in df.columns:
        df['Odometer'] = (
            df['Odometer']
            .astype(str)
            .str.replace(',', '', regex=False)
            .astype(float)
        )

    # --- Feature engineering ---
    df['Age'] = 2025 - df['Year']                              # vehicle age
    df['km_per_year'] = df['Odometer'] / df['Age']             # average annual mileage
    # normalize skewed numeric features
    for col in ['Engine', 'Odometer', 'Age', 'km_per_year']:
        df[col] = np.log1p(df[col])

    return df


# ----------------------------- #
# 2. Main training routine      #
# ----------------------------- #
def build_and_save(df: pd.DataFrame):
    y = df['Price']
    X = df.drop(columns=['Price'])

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- preprocessing pipelines with imputation ---
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    pipe = Pipeline(steps=[('pre', pre), ('model', model)])

    # --- K-Fold Cross-Validation with early stopping ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(
            X_tr, y_tr,
            model__eval_set=[(pipe.named_steps['pre'].transform(X_va), y_va)],
            model__early_stopping_rounds=50,
            model__verbose=False
        )
        preds_va = pipe.predict(X_va)
        rmse_va = mean_squared_error(y_va, preds_va, squared=False)
        cv_rmses.append(rmse_va)
    print(f"CV RMSE: {np.mean(cv_rmses):,.2f} ± {np.std(cv_rmses):,.2f}")
    # Refit on full dataset after CV
    pipe.fit(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(
        X_train, y_train,
        model__eval_set=[(pipe.named_steps['pre'].transform(X_val), y_val)],
        model__early_stopping_rounds=50,
        model__verbose=False
    )
    preds = pipe.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    # Handle both new and old scikit‑learn signatures
    if 'squared' in signature(mean_squared_error).parameters:
        rmse = mean_squared_error(y_val, preds, squared=False)
    else:
        rmse = mean_squared_error(y_val, preds) ** 0.5
    print(f"Validation MAE: {mae:,.2f}")
    print(f"Validation RMSE: {rmse:,.2f}")

    # Build similarity index using preprocessed full dataset
    X_all_transformed = pipe.named_steps['pre'].transform(X)
    # Convert to dense if sparse and ensure no remaining NaNs
    if hasattr(X_all_transformed, "toarray"):
        X_all_transformed = X_all_transformed.toarray()
    X_all_transformed = np.nan_to_num(X_all_transformed)

    nn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
    nn.fit(X_all_transformed)

    # Persist artifacts
    joblib.dump(pipe, 'model_price.pkl')
    joblib.dump(nn, 'model_similar.pkl')
    df.to_pickle('dataset_clean.pkl')
    print('Artifacts saved: model_price.pkl, model_similar.pkl, dataset_clean.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        type=str,
        default='data/vehicle_prices.csv',
        help='Path to the raw vehicle CSV downloaded from Kaggle'
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV file not found at {args.csv}. "
            "Download it from Kaggle and place it accordingly.")

    df_raw = pd.read_csv(args.csv)
    df = clean_dataframe(df_raw)
    build_and_save(df)