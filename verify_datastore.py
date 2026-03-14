# verify_datastore.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-mlops-loan-approval",
    workspace_name="mlw-loan-approval"
)

# List all datastores in the workspace
for ds in ml_client.datastores.list():
    print(f"Name: {ds.name}, Type: {ds.type}")

# Get the default datastore (blob storage)
default_ds = ml_client.datastores.get_default()
print(f"\nDefault Datastore: {default_ds.name}")
print(f"Storage Account: {default_ds.account_name}")
print(f"Container: {default_ds.container_name}")
