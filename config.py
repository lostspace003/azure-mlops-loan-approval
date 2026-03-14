# config.py
import os
from pathlib import Path

# Load .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_azure_config():
    return {
        'subscription_id': os.environ.get('SUBSCRIPTION_ID', '2f55ca6d-7f38-4220-919c-591a66eb1a2b'),
        'resource_group': os.environ.get('RESOURCE_GROUP', 'rg-mlops-loan-approval'),
        'workspace_name': os.environ.get('WORKSPACE_NAME', 'mlw-loan-approval-v2'),
    }


def get_ml_client():
    cfg = get_azure_config()
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=cfg['subscription_id'],
        resource_group_name=cfg['resource_group'],
        workspace_name=cfg['workspace_name'],
    )


def get_sql_connection_string():
    server = os.environ.get('SQL_SERVER', 'sqlsrv-mlops-loan.database.windows.net')
    database = os.environ.get('SQL_DATABASE', 'sqldb-loan-data')
    username = os.environ.get('SQL_USERNAME', 'sqladmin')
    password = os.environ.get('SQL_PASSWORD', 'MLOps@2026Secure!')
    return (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={username};"
        f"Pwd={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
