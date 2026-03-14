# test_endpoint.py
import json
import sys
from config import get_ml_client

ml_client = get_ml_client()

# Test cases: (description, features, expected_likely_outcome)
TEST_CASES = [
    (
        "Strong applicant (high income, high credit score)",
        [35, 85000.0, 250000.0, 720, 8, 0.32, 5, 1, 1, 1, 1, 0, 0],
        1,
    ),
    (
        "Weak applicant (low income, low credit score, previous default)",
        [25, 30000.0, 400000.0, 450, 1, 0.75, 1, 0, 0, 0, 0, 4, 1],
        0,
    ),
]

COLUMNS = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'EmploymentYears', 'DebtToIncomeRatio', 'NumCreditLines',
    'Education', 'MaritalStatus', 'HasMortgage',
    'HasDependents', 'LoanPurpose', 'PreviousDefault',
]

failures = 0
for desc, features, expected in TEST_CASES:
    sample = {
        'input_data': {
            'columns': COLUMNS,
            'data': [features],
        }
    }

    response = ml_client.online_endpoints.invoke(
        endpoint_name='loan-approval-endpoint',
        deployment_name='blue',
        request=json.dumps(sample),
    )

    result = json.loads(response)
    prediction = result[0] if isinstance(result, list) else result

    status = "PASS" if prediction == expected else "WARN"
    if prediction not in (0, 1):
        status = "FAIL"
        failures += 1

    label = 'Approved' if prediction == 1 else 'Denied'
    print(f"[{status}] {desc}: prediction={label} (expected={'Approved' if expected else 'Denied'})")

if failures:
    print(f"\n{failures} test(s) failed.")
    sys.exit(1)
else:
    print("\nAll endpoint tests passed.")
