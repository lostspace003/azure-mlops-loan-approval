# verify_workspace.py
from config import get_ml_client, get_azure_config

ml_client = get_ml_client()
cfg = get_azure_config()

workspace = ml_client.workspaces.get(cfg['workspace_name'])
print(f"Workspace: {workspace.name}")
print(f"Location: {workspace.location}")
print(f"Resource Group: {workspace.resource_group}")
