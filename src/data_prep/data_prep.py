# src/data_prep/data_prep.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import mlflow


CATEGORICAL_COLUMNS = ['Education', 'MaritalStatus', 'LoanPurpose']
REQUIRED_COLUMNS = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears',
    'DebtToIncomeRatio', 'NumCreditLines', 'Education', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'PreviousDefault',
    'Approved'
]


def validate_data(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) > 0:
        print(f"Warning: null values found:\n{nulls}")
        df = df.dropna(subset=REQUIRED_COLUMNS)
        print(f"Rows after dropping nulls: {len(df)}")

    if not set(df['Approved'].unique()).issubset({0, 1}):
        raise ValueError("Target column 'Approved' must be binary (0/1)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    mlflow.start_run()

    # Read and validate data
    df = pd.read_csv(args.input_data)
    print(f"Input shape: {df.shape}")

    df = df.drop('ApplicationID', axis=1, errors='ignore')
    df = validate_data(df)

    # Encode categorical features and save mappings
    le_dict = {}
    label_mappings = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        label_mappings[col] = {
            str(cls): int(idx)
            for idx, cls in enumerate(le.classes_)
        }

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42,
        stratify=df['Approved']
    )

    # Log metrics
    mlflow.log_metric('train_samples', len(train_df))
    mlflow.log_metric('test_samples', len(test_df))
    mlflow.log_metric('num_features', len(df.columns) - 1)
    mlflow.log_metric('approval_rate', df['Approved'].mean())
    mlflow.log_dict(label_mappings, 'label_encoders.json')

    # Save outputs
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.test_data, 'test.csv'), index=False)

    # Save label encoder mappings alongside train data for inference
    with open(os.path.join(args.train_data, 'label_encoders.json'), 'w') as f:
        json.dump(label_mappings, f, indent=2)

    mlflow.end_run()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == '__main__':
    main()
