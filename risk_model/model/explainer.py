import shap
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap_values(model, X_train, entity_features, top_n=10, plot=True):
    """
    Computes SHAP values for a single entity and plots the feature contributions.
    - model: trained XGBClassifier
    - X_train: training features used to train the model
    - entity_features: features of the entity you want to explain (as a DataFrame with one row)
    - top_n: number of top features to show
    """
    # Create explainer using training data
    explainer = shap.Explainer(model, X_train, model_output="probability")
    
    # Compute SHAP values for the entity
    shap_values = explainer(entity_features)
    # shap_values = explainer.shap_values(entity_features)
    
    # # For binary classification, shap_values is a list of two arrays (one for each class)
    # # We pick class 1 (the risk class)
    # shap_values_for_class1 = shap_values[1]
    
    # Prepare a DataFrame of feature contributions
    contributions = pd.DataFrame({
        "Feature": entity_features.columns,
        "SHAP Value": shap_values.values[0]
        # "SHAP Value": shap_values_for_class1[0]
    })
    contributions["Abs SHAP Value"] = contributions["SHAP Value"].abs()
    
    # Select top N features by absolute SHAP value
    top_features = contributions.sort_values("Abs SHAP Value", ascending=False).head(top_n)
    
    # Plot SHAP values if requested
    if plot:
        # # Bar plot for top features of the single entity
        # plt.figure(figsize=(8, 6))
        # plt.barh(top_features["Feature"][::-1], top_features["SHAP Value"][::-1])
        # plt.xlabel("SHAP Value")
        # plt.title("Top Feature Contributions for Entity")
        # plt.show()
        
        # Plot
        # shap.plots.bar(shap_values_for_class1, max_display=top_n, show=True)
        shap.plots.bar(shap_values, max_display=top_n, show=True)
    
    return top_features


# def compute_shap_values(model, X_train, entity_features, top_n):
#     explainer = shap.Explainer(model, X_train)
#     shap_values = explainer(entity_features)
#     contributions = pd.DataFrame({
#         "Feature": entity_features.columns,
#         "SHAP Value": shap_values.values[0]
#     })
#     contributions["Abs SHAP Value"] = abs(contributions["SHAP Value"])
#     top_features = contributions.sort_values("Abs SHAP Value", ascending=False).head(top_n)
#     return top_features