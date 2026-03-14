# tests/test_data_prep.py
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_sample_data(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(21, 65, n),
        'Income': np.random.uniform(25000, 200000, n),
        'LoanAmount': np.random.uniform(5000, 500000, n),
        'CreditScore': np.random.randint(300, 850, n),
        'EmploymentYears': np.random.randint(0, 40, n),
        'DebtToIncomeRatio': np.random.uniform(0.05, 0.80, n),
        'NumCreditLines': np.random.randint(0, 15, n),
        'Education': np.random.randint(0, 4, n),
        'MaritalStatus': np.random.randint(0, 3, n),
        'HasMortgage': np.random.randint(0, 2, n),
        'HasDependents': np.random.randint(0, 2, n),
        'LoanPurpose': np.random.randint(0, 5, n),
        'PreviousDefault': np.random.randint(0, 2, n),
        'Approved': np.random.randint(0, 2, n)
    })


def test_data_shape():
    df = create_sample_data()
    assert df.shape == (100, 14)


def test_train_test_split_ratio():
    df = create_sample_data()
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    assert len(train) == 80
    assert len(test) == 20


def test_no_null_values():
    df = create_sample_data()
    assert df.isnull().sum().sum() == 0


def test_target_is_binary():
    df = create_sample_data()
    assert set(df['Approved'].unique()).issubset({0, 1})
