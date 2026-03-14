# src/evaluate/evaluate.py
import argparse, os, json
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--evaluation_output', type=str, required=True)
    parser.add_argument('--accuracy_threshold', type=float, default=0.75)
    args = parser.parse_args()

    mlflow.start_run()

    # Load model and test data
    model = mlflow.sklearn.load_model(args.model_input)
    test_df = pd.read_csv(os.path.join(args.test_data, 'test.csv'))
    X_test = test_df.drop('Approved', axis=1)
    y_test = test_df['Approved']

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'passed_threshold': accuracy_score(y_test, y_pred) >= args.accuracy_threshold
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v if isinstance(v, (int, float)) else int(v))

    # Save evaluation results
    os.makedirs(args.evaluation_output, exist_ok=True)
    with open(os.path.join(args.evaluation_output, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    mlflow.end_run()
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Passed threshold: {metrics['passed_threshold']}")


if __name__ == '__main__':
    main()
