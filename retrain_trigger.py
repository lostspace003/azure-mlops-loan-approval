# retrain_trigger.py
import requests, json, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ['SUBSCRIPTION_ID'],
    resource_group_name=os.environ['RESOURCE_GROUP'],
    workspace_name=os.environ['WORKSPACE_NAME']
)

# Check latest monitoring run
monitor = ml_client.schedules.get('loan-approval-monitor')
print(f"Monitor status: {monitor.provisioning_state}")

# If drift detected, trigger DevOps pipeline
DRIFT_THRESHOLD = 0.3  # Normalized Wasserstein distance


def trigger_retrain_pipeline():
    """Trigger Azure DevOps CI pipeline via REST API"""
    org = 'your-org'
    project = 'mlops-loan-approval'
    pipeline_id = '<ci-pipeline-id>'
    pat = os.environ['AZURE_DEVOPS_PAT']

    url = (f'https://dev.azure.com/{org}/{project}'
           f'/_apis/pipelines/{pipeline_id}/runs?api-version=7.1')

    response = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        auth=('', pat),
        json={
            'resources': {
                'repositories': {
                    'self': {'refName': 'refs/heads/main'}
                }
            },
            'variables': {
                'RETRAIN_REASON': {'value': 'data_drift_detected'}
            }
        }
    )

    if response.status_code == 200:
        print(f"Retrain pipeline triggered: {response.json()['id']}")
    else:
        print(f"Failed to trigger: {response.text}")


# Example: trigger based on drift detection
# In production, this would be called from the monitoring alert
trigger_retrain_pipeline()
