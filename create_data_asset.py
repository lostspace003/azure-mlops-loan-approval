# create_data_asset.py
import pyodbc
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Extract data from SQL to CSV
conn_str = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=tcp:<your-sql-server>.database.windows.net,1433;"
    "Database=sqldb-loan-data;"
    "Uid=sqladmin;"
    "Pwd=MLOps@2026Secure!;"
    "Encrypt=yes;"
    "TrustServerCertificate=False;"
    "Connection Timeout=30;"
)

conn = pyodbc.connect(conn_str)
df = pd.read_sql("SELECT * FROM LoanApplications", conn)
conn.close()

# Save locally
df.to_csv("./data/loan_applications.csv", index=False)
print(f"Extracted {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution:\n{df['Approved'].value_counts()}")

# Register as AML Data Asset
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-mlops-loan-approval",
    workspace_name="mlw-loan-approval"
)

loan_data = Data(
    name="loan-approval-dataset",
    description="BFSI Loan Approval data - 1000 rows, 14 features",
    path="./data/loan_applications.csv",
    type=AssetTypes.URI_FILE,
    version="1"
)

ml_client.data.create_or_update(loan_data)
print("Data asset 'loan-approval-dataset' v1 registered.")
