# verify_datastore.py
from config import get_ml_client

ml_client = get_ml_client()

# List all datastores in the workspace
for ds in ml_client.datastores.list():
    print(f"Name: {ds.name}, Type: {ds.type}")

# Get the default datastore (blob storage)
default_ds = ml_client.datastores.get_default()
print(f"\nDefault Datastore: {default_ds.name}")
print(f"Storage Account: {default_ds.account_name}")
print(f"Container: {default_ds.container_name}")
