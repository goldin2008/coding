import shap
import pandas as pd
import matplotlib.pyplot as plt
from config import TOP_N_FEATURES

def compute_shap_values(model, X_train, entity_features, top_n=TOP_N_FEATURES, plot=True):
    """
    Computes SHAP values for a single entity and plots the feature contributions.

    Parameters:
    - model: Trained XGBClassifier
    - X_train: Training features used to train the model
    - entity_features: Single-row DataFrame of the entity’s features
    - top_n: Number of top features to show (default from config)
    - plot: Whether to display a SHAP bar plot

    Returns:
    - top_features: DataFrame of top features and their SHAP values
    """
    explainer = shap.Explainer(model, X_train, model_output="probability")
    shap_values = explainer(entity_features)

    contributions = pd.DataFrame({
        "Feature": entity_features.columns,
        "SHAP Value": shap_values.values[0]
    })
    contributions["Abs SHAP Value"] = contributions["SHAP Value"].abs()

    top_features = contributions.sort_values("Abs SHAP Value", ascending=False).head(top_n)

    if plot:
        shap.plots.bar(shap_values, max_display=top_n, show=True)

    return top_features


def generate_shap_summary(model, X, plot_summary=True):
    """
    Generate and optionally plot SHAP values for a given model and dataset.

    Parameters:
    - model: Trained model (e.g., XGBClassifier)
    - X: Feature matrix to explain
    - plot_summary: Whether to display a summary plot

    Returns:
    - explainer: SHAP TreeExplainer object
    - shap_values: Computed SHAP values
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if plot_summary:
        shap.summary_plot(shap_values, X)

    return explainer, shap_values
