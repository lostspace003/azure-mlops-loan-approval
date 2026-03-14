# deploy_endpoint.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-mlops-loan-approval",
    workspace_name="mlw-loan-approval"
)

# Create endpoint
endpoint_name = "loan-approval-endpoint"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Real-time loan approval predictions",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Endpoint '{endpoint_name}' created.")

# Get latest registered model
model = ml_client.models.get('loan-approval-model', label='latest')
print(f"Model: {model.name} v{model.version}")

# Create deployment (no-code: no score.py or environment needed)
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route 100% traffic to blue deployment
endpoint.traffic = {'blue': 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Deployment complete. 100% traffic routed to 'blue'.")
