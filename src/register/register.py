# src/register/register.py
import argparse, os, json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str, required=True)
    parser.add_argument('--evaluation_output', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='loan-approval-model')
    args = parser.parse_args()

    # Check if model passed threshold
    metrics_path = os.path.join(args.evaluation_output, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if not metrics.get('passed_threshold', False):
        print('Model did NOT pass accuracy threshold. Skipping registration.')
        return

    # Register model
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ.get('SUBSCRIPTION_ID'),
        resource_group_name=os.environ.get('RESOURCE_GROUP'),
        workspace_name=os.environ.get('WORKSPACE_NAME')
    )

    model = Model(
        path=args.model_input,
        name=args.model_name,
        description=f"Loan approval classifier. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}",
        type=AssetTypes.MLFLOW_MODEL,
        tags={
            'accuracy': str(round(metrics['accuracy'], 4)),
            'auc_roc': str(round(metrics['auc_roc'], 4)),
            'f1_score': str(round(metrics['f1_score'], 4)),
            'framework': 'scikit-learn',
            'task': 'classification'
        }
    )

    registered = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered.name} v{registered.version}")


if __name__ == '__main__':
    main()
