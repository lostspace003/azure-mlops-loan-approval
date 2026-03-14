# tests/test_evaluate.py
import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class TestMetrics:
    def setup_method(self):
        np.random.seed(42)
        self.y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        self.y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])
        self.y_prob = np.array([0.9, 0.8, 0.4, 0.2, 0.1, 0.6, 0.7, 0.3, 0.85, 0.15])

    def test_accuracy(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        assert 0 <= acc <= 1
        assert acc == 0.8  # 8/10 correct

    def test_precision(self):
        prec = precision_score(self.y_true, self.y_pred)
        assert 0 <= prec <= 1

    def test_recall(self):
        rec = recall_score(self.y_true, self.y_pred)
        assert 0 <= rec <= 1

    def test_f1_between_precision_and_recall(self):
        prec = precision_score(self.y_true, self.y_pred)
        rec = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        assert min(prec, rec) <= f1 <= max(prec, rec)

    def test_auc_roc(self):
        auc = roc_auc_score(self.y_true, self.y_prob)
        assert 0 <= auc <= 1
        assert auc > 0.5  # Better than random

    def test_confusion_matrix_shape(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_sums_to_total(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        assert cm.sum() == len(self.y_true)


class TestThreshold:
    def test_passes_threshold(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])  # Perfect
        acc = accuracy_score(y_true, y_pred)
        assert acc >= 0.75

    def test_fails_threshold(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])  # All wrong
        acc = accuracy_score(y_true, y_pred)
        assert acc < 0.75


class TestEdgeCases:
    def test_all_same_predictions(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.6

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        assert accuracy_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred) == 1.0
