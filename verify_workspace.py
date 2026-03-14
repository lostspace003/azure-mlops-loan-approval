# verify_workspace.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-mlops-loan-approval",
    workspace_name="mlw-loan-approval"
)

workspace = ml_client.workspaces.get("mlw-loan-approval")
print(f"Workspace: {workspace.name}")
print(f"Location: {workspace.location}")
print(f"Resource Group: {workspace.resource_group}")
