"""SHAP helpers shared by the analysis script and the backend."""
import numpy as np


def class_shap(sv, class_idx):
    """Return an (n_samples, n_features) array for one class, for any shap version."""
    if isinstance(sv, list):
        return np.asarray(sv[class_idx])
    sv = np.asarray(sv)
    return sv[:, :, class_idx] if sv.ndim == 3 else sv


def explainer_for(model):
    import shap
    return shap.TreeExplainer(model)


def top_contributions(explainer, x_row, class_idx):
    """
    x_row: single-row DataFrame. Returns ALL features sorted by |SHAP| (largest first):
    [{"feature", "value", "shap"}], where shap is the push toward `class_idx`.
    """
    vals = class_shap(explainer.shap_values(x_row), class_idx)[0]
    out = [{"feature": c, "value": float(x_row.iloc[0][c]), "shap": float(v)}
           for c, v in zip(x_row.columns, vals)]
    return sorted(out, key=lambda d: abs(d["shap"]), reverse=True)
