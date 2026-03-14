# retrain_trigger.py
import os
import sys
import requests
import json
from config import get_ml_client

ml_client = get_ml_client()

DRIFT_THRESHOLD = 0.3  # Normalized Wasserstein distance


def get_latest_drift_metric():
    """Check latest monitoring run for drift."""
    monitor = ml_client.schedules.get('loan-approval-monitor')
    print(f"Monitor status: {monitor.provisioning_state}")

    # List recent monitor runs and check for drift signals
    jobs = ml_client.jobs.list(
        parent_job_name=None,
        experiment_name='loan-approval-monitor',
    )
    for job in jobs:
        if job.status == 'Completed':
            print(f"Latest monitor run: {job.name} ({job.status})")
            # In production, parse the drift metric from job outputs
            # Return the drift score if available
            return None
    return None


def trigger_retrain_pipeline(reason="data_drift_detected"):
    """Trigger Azure DevOps CI pipeline via REST API."""
    org = os.environ.get('AZURE_DEVOPS_ORG', 'your-org')
    project = os.environ.get('AZURE_DEVOPS_PROJECT', 'mlops-loan-approval')
    pipeline_id = os.environ.get('AZURE_DEVOPS_PIPELINE_ID')
    pat = os.environ.get('AZURE_DEVOPS_PAT')

    if not pat or not pipeline_id:
        print("Error: AZURE_DEVOPS_PAT and AZURE_DEVOPS_PIPELINE_ID must be set.")
        sys.exit(1)

    url = (
        f'https://dev.azure.com/{org}/{project}'
        f'/_apis/pipelines/{pipeline_id}/runs?api-version=7.1'
    )

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
                'RETRAIN_REASON': {'value': reason}
            },
        },
        timeout=30,
    )

    if response.status_code == 200:
        run_id = response.json().get('id')
        print(f"Retrain pipeline triggered: run {run_id}")
    else:
        print(f"Failed to trigger pipeline ({response.status_code}): {response.text}")
        sys.exit(1)


if __name__ == '__main__':
    drift_score = get_latest_drift_metric()

    if drift_score is not None and drift_score > DRIFT_THRESHOLD:
        print(f"Drift detected ({drift_score:.3f} > {DRIFT_THRESHOLD}). Triggering retrain.")
        trigger_retrain_pipeline(reason=f"drift_{drift_score:.3f}")
    elif drift_score is not None:
        print(f"Drift within threshold ({drift_score:.3f} <= {DRIFT_THRESHOLD}). No retrain needed.")
    else:
        print("No drift metric available. Check monitor runs in Azure ML Studio.")
