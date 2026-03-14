# create_data_asset.py
import os
import pyodbc
import pandas as pd
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from config import get_ml_client, get_sql_connection_string

# Extract data from SQL to CSV
conn_str = get_sql_connection_string()
conn = pyodbc.connect(conn_str)
df = pd.read_sql("SELECT * FROM LoanApplications", conn)
conn.close()

# Save locally
os.makedirs("./data", exist_ok=True)
df.to_csv("./data/loan_applications.csv", index=False)
print(f"Extracted {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution:\n{df['Approved'].value_counts()}")

# Register as AML Data Asset
ml_client = get_ml_client()

loan_data = Data(
    name="loan-approval-dataset",
    description="BFSI Loan Approval data - 1000 rows, 14 features",
    path="./data/loan_applications.csv",
    type=AssetTypes.URI_FILE,
    version="1"
)

ml_client.data.create_or_update(loan_data)
print("Data asset 'loan-approval-dataset' v1 registered.")
