import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import SHAP_CONFIG


def compute_shap_values(model, X_train, entity_features, top_n=None, plot=None):
    """
    Computes SHAP values for one or more entities and optionally plots feature contributions.

    Parameters:
    - model: Trained XGBClassifier
    - X_train: Training features used to train the model
    - entity_features: DataFrame of one or more entities' features
    - top_n: Number of top features to show (default from config)
    - plot: Whether to display a SHAP bar plot (default from config)

    Returns:
    - top_features_df: DataFrame with 'Feature', 'Mean Abs SHAP Value', 'SHAP Value'
    - shap_values: SHAP values object
    """
    # Fallback to config values if not explicitly provided
    top_n = top_n if top_n is not None else SHAP_CONFIG.get("top_n_features", 10)
    plot = plot if plot is not None else SHAP_CONFIG.get("plot_shap_summary", True)

    explainer = shap.Explainer(model, X_train, model_output="probability")
    shap_values = explainer(entity_features)

    feature_names = entity_features.columns

    if entity_features.shape[0] == 1:
        # Single entity
        contributions = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values.values[0]
        })
        contributions["Mean Abs SHAP Value"] = contributions["SHAP Value"].abs()
        top_features_df = contributions.sort_values("Mean Abs SHAP Value", ascending=False).head(top_n)
    else:
        # Multiple entities
        abs_mean = np.abs(shap_values.values).mean(axis=0)
        signed_mean = shap_values.values.mean(axis=0)
        
        top_features_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Abs SHAP Value": abs_mean,
            "SHAP Value": signed_mean
        }).sort_values("Mean Abs SHAP Value", ascending=False).head(top_n).reset_index(drop=True)

    if plot:
        shap.plots.bar(shap_values, max_display=top_n, show=True)

    return top_features_df, shap_values


def generate_shap_summary(model, X, plot_summary=None):
    """
    Generate and optionally plot SHAP values for a given model and dataset.

    Parameters:
    - model: Trained model (e.g., XGBClassifier)
    - X: Feature matrix to explain
    - plot_summary: Whether to display a summary plot (default from config)

    Returns:
    - explainer: SHAP TreeExplainer object
    - shap_values: Computed SHAP values
    """
    plot_summary = plot_summary if plot_summary is not None else SHAP_CONFIG.get("plot_shap_summary", True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if plot_summary:
        shap.summary_plot(shap_values, X)

    # return explainer, shap_values
