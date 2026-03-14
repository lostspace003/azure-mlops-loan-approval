# tests/test_train.py
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def create_training_data(n=200):
    np.random.seed(42)
    X = pd.DataFrame({
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
    })
    # Simple approval logic for testing
    y = ((X['CreditScore'] > 600) & (X['Income'] > 50000)).astype(int)
    return X, y


class TestModelTraining:
    def test_model_fits_without_error(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        assert hasattr(model, 'feature_importances_')

    def test_model_produces_binary_predictions(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_produces_probabilities(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_training_accuracy_above_random(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        assert acc > 0.5

    def test_feature_importance_sums_to_one(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6

    def test_feature_importance_count_matches_features(self):
        X, y = create_training_data()
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        assert len(model.feature_importances_) == X.shape[1]


class TestHyperparameters:
    def test_more_estimators_improves_training(self):
        X, y = create_training_data()
        model_small = GradientBoostingClassifier(
            n_estimators=5, max_depth=3, random_state=42
        )
        model_large = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        model_small.fit(X, y)
        model_large.fit(X, y)
        acc_small = accuracy_score(y, model_small.predict(X))
        acc_large = accuracy_score(y, model_large.predict(X))
        assert acc_large >= acc_small
