# tests/test_data_prep.py
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_sample_data(n=100):
    """Create sample data with realistic categorical values (pre-encoding)."""
    np.random.seed(42)
    return pd.DataFrame({
        'ApplicationID': range(1, n + 1),
        'Age': np.random.randint(21, 65, n),
        'Income': np.random.uniform(25000, 200000, n),
        'LoanAmount': np.random.uniform(5000, 500000, n),
        'CreditScore': np.random.randint(300, 850, n),
        'EmploymentYears': np.random.randint(0, 40, n),
        'DebtToIncomeRatio': np.random.uniform(0.05, 0.80, n),
        'NumCreditLines': np.random.randint(0, 15, n),
        'Education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
        'HasMortgage': np.random.randint(0, 2, n),
        'HasDependents': np.random.randint(0, 2, n),
        'LoanPurpose': np.random.choice(['Home', 'Car', 'Education', 'Business', 'Personal'], n),
        'PreviousDefault': np.random.randint(0, 2, n),
        'Approved': np.random.randint(0, 2, n),
    })


def create_encoded_data(n=100):
    """Create sample data with already-encoded values (post-encoding)."""
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
        'Approved': np.random.randint(0, 2, n),
    })


class TestDataShape:
    def test_raw_data_has_correct_columns(self):
        df = create_sample_data()
        assert 'ApplicationID' in df.columns
        assert 'Approved' in df.columns
        assert df.shape == (100, 15)

    def test_encoded_data_shape(self):
        df = create_encoded_data()
        assert df.shape == (100, 14)

    def test_no_null_values(self):
        df = create_sample_data()
        assert df.isnull().sum().sum() == 0


class TestTrainTestSplit:
    def test_split_ratio_default(self):
        df = create_encoded_data()
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_split_ratio_custom(self):
        df = create_encoded_data()
        train, test = train_test_split(df, test_size=0.3, random_state=42)
        assert len(train) == 70
        assert len(test) == 30

    def test_stratified_split_preserves_ratio(self):
        df = create_encoded_data()
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['Approved']
        )
        original_rate = df['Approved'].mean()
        train_rate = train['Approved'].mean()
        assert abs(original_rate - train_rate) < 0.05


class TestTargetColumn:
    def test_target_is_binary(self):
        df = create_sample_data()
        assert set(df['Approved'].unique()).issubset({0, 1})

    def test_target_has_both_classes(self):
        df = create_sample_data(n=500)
        assert 0 in df['Approved'].values
        assert 1 in df['Approved'].values


class TestCategoricalEncoding:
    def test_education_categories(self):
        df = create_sample_data()
        expected = {'HighSchool', 'Bachelor', 'Master', 'PhD'}
        assert set(df['Education'].unique()).issubset(expected)

    def test_marital_status_categories(self):
        df = create_sample_data()
        expected = {'Single', 'Married', 'Divorced'}
        assert set(df['MaritalStatus'].unique()).issubset(expected)

    def test_loan_purpose_categories(self):
        df = create_sample_data()
        expected = {'Home', 'Car', 'Education', 'Business', 'Personal'}
        assert set(df['LoanPurpose'].unique()).issubset(expected)

    def test_label_encoding_produces_integers(self):
        from sklearn.preprocessing import LabelEncoder
        df = create_sample_data()
        le = LabelEncoder()
        encoded = le.fit_transform(df['Education'])
        assert encoded.dtype in [np.int32, np.int64]
        assert len(le.classes_) == len(df['Education'].unique())


class TestFeatureRanges:
    def test_age_range(self):
        df = create_sample_data()
        assert df['Age'].min() >= 21
        assert df['Age'].max() <= 65

    def test_credit_score_range(self):
        df = create_sample_data()
        assert df['CreditScore'].min() >= 300
        assert df['CreditScore'].max() <= 850

    def test_income_positive(self):
        df = create_sample_data()
        assert (df['Income'] > 0).all()

    def test_dti_ratio_range(self):
        df = create_sample_data()
        assert df['DebtToIncomeRatio'].min() >= 0
        assert df['DebtToIncomeRatio'].max() <= 1.0


class TestDataValidation:
    def test_missing_column_raises_error(self):
        from src.data_prep.data_prep import validate_data
        df = create_encoded_data()
        df = df.drop('CreditScore', axis=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_valid_data_passes(self):
        from src.data_prep.data_prep import validate_data
        df = create_encoded_data()
        result = validate_data(df)
        assert len(result) == len(df)

    def test_nulls_are_dropped(self):
        from src.data_prep.data_prep import validate_data
        df = create_encoded_data()
        df.loc[0, 'Income'] = np.nan
        df.loc[1, 'Age'] = np.nan
        result = validate_data(df)
        assert len(result) == 98
        assert result.isnull().sum().sum() == 0
