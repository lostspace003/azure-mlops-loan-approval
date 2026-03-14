# src/train/train.py
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report)
import mlflow, mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--model_output', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    args = parser.parse_args()

    mlflow.start_run()
    mlflow.autolog()

    # Load training data
    train_df = pd.read_csv(os.path.join(args.train_data, 'train.csv'))
    X_train = train_df.drop('Approved', axis=1)
    y_train = train_df['Approved']

    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Log training accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    mlflow.log_metric('train_accuracy', train_acc)

    # Log feature importance
    importance = dict(zip(X_train.columns, model.feature_importances_))
    mlflow.log_dict(importance, 'feature_importance.json')

    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)

    mlflow.end_run()
    print(f"Training accuracy: {train_acc:.4f}")


if __name__ == '__main__':
    main()
