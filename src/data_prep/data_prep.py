# src/data_prep/data_prep.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    mlflow.start_run()

    # Read data
    df = pd.read_csv(args.input_data)
    print(f"Input shape: {df.shape}")

    # Drop ID column
    df = df.drop('ApplicationID', axis=1, errors='ignore')

    # Encode categorical features
    le_dict = {}
    for col in ['Education', 'MaritalStatus', 'LoanPurpose']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

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

    # Save outputs
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.test_data, 'test.csv'), index=False)

    mlflow.end_run()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == '__main__':
    main()
