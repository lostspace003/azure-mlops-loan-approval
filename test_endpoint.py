# test_endpoint.py
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<your-subscription-id>",
    resource_group_name="rg-mlops-loan-approval",
    workspace_name="mlw-loan-approval"
)

# Sample loan application - MLflow format requires column names
sample = {
    'input_data': {
        'columns': [
            'Age', 'Income', 'LoanAmount', 'CreditScore',
            'EmploymentYears', 'DebtToIncomeRatio', 'NumCreditLines',
            'Education', 'MaritalStatus', 'HasMortgage',
            'HasDependents', 'LoanPurpose', 'PreviousDefault'
        ],
        'data': [[
            35, 85000.0, 250000.0, 720, 8, 0.32, 5,
            1, 1, 1, 1, 0, 0
        ]]
    }
}

response = ml_client.online_endpoints.invoke(
    endpoint_name='loan-approval-endpoint',
    deployment_name='blue',
    request=json.dumps(sample)
)

result = json.loads(response)
print(f"Raw response: {result}")
# MLflow returns predictions as a list, e.g. [1] or [0]
prediction = result[0] if isinstance(result, list) else result
print(f"Prediction: {'Approved' if prediction == 1 else 'Denied'}")
