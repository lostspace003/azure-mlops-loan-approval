# src/register/register.py
import argparse
import os
import json
import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str, required=True)
    parser.add_argument('--evaluation_output', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='loan-approval-model')
    args = parser.parse_args()

    # Check if model passed threshold
    metrics_path = os.path.join(args.evaluation_output, 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Error: metrics file not found at {metrics_path}")
        raise FileNotFoundError(metrics_path)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if not metrics.get('passed_threshold', False):
        print('Model did NOT pass accuracy threshold. Skipping registration.')
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
        return

    # Register model using MLflow (works on AML compute without explicit credentials)
    mlflow.start_run()

    model_uri = f"file://{os.path.abspath(args.model_input)}"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name,
        tags={
            'accuracy': str(round(metrics['accuracy'], 4)),
            'auc_roc': str(round(metrics['auc_roc'], 4)),
            'f1_score': str(round(metrics['f1_score'], 4)),
            'framework': 'scikit-learn',
            'task': 'classification',
        },
    )

    mlflow.end_run()
    print(f"Model registered: {result.name} v{result.version}")


if __name__ == '__main__':
    main()
