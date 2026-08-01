"""Evaluation helpers."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def classification_details(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
