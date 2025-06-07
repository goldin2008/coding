# shap_analysis.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import config  # Import config


def summarize_top_features(shap_values, X_test, top_n=None):
    """
    Summarizes the most frequently occurring top features across all samples
    based on their absolute SHAP values.

    Parameters:
    - shap_values: SHAP values object from SHAP package
    - X_test: DataFrame of features
    - top_n: Number of top features to consider per sample (overrides config if provided)

    Returns:
    - summary_df: DataFrame with feature names and their frequencies
    """
    if top_n is None:
        top_n = config.SHAP_CONFIG["top_n_features"]

    feature_names = X_test.columns
    feature_counts = {}

    for i in range(len(shap_values.values)):
        sample_shap = shap_values.values[i]
        top_features = pd.Series(sample_shap, index=feature_names).abs().nlargest(top_n).index
        for feature in top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    summary_df = pd.DataFrame(list(feature_counts.items()), columns=["Feature", "Frequency"])
    summary_df.sort_values("Frequency", ascending=False, inplace=True)
    return summary_df


def identify_high_impact_features(shap_values, X_test):
    """
    Identifies consistent high-impact features by calculating the mean absolute
    SHAP value for each feature across all samples.

    Parameters:
    - shap_values: SHAP values object from SHAP package
    - X_test: DataFrame of features

    Returns:
    - mean_abs_shap: Series of features with their mean absolute SHAP values
    """
    feature_names = X_test.columns
    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)
    return mean_abs_shap


def cluster_features(mean_abs_shap, n_clusters=None):
    """
    Clusters features based on their mean absolute SHAP values.

    Parameters:
    - mean_abs_shap: Series of mean absolute SHAP values per feature
    - n_clusters: Number of clusters to form (overrides config if provided)

    Returns:
    - clustered: DataFrame with features, mean SHAP values, and cluster labels
    """
    if n_clusters is None:
        n_clusters = config.CLUSTER_CONFIG["n_clusters"]

    random_state = config.CLUSTER_CONFIG.get("random_state", 42)
    shap_matrix = mean_abs_shap.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(shap_matrix)
    clustered = pd.DataFrame({
        "Feature": mean_abs_shap.index,
        "Mean Abs SHAP": mean_abs_shap.values,
        "Cluster": cluster_labels
    }).sort_values("Cluster")
    return clustered


def build_llm_prompt_for_hypotheses(top_features):
    """
    Builds a prompt asking an LLM to hypothesize why the top features might
    influence risk scores and suggest business drivers.

    Parameters:
    - top_features: List of top feature names

    Returns:
    - prompt: String prompt for LLM
    """
    feature_list = "\n".join([f"- {feature}" for feature in top_features])
    prompt = (
        "You are a risk model explanation assistant. Given the following features that "
        "frequently contribute to high risk scores:\n\n"
        f"{feature_list}\n\n"
        "For each feature:\n"
        "- Explain why it might influence the risk score in an AML context.\n"
        "- Suggest potential data quality issues or business processes that might cause this feature to be flagged.\n"
    )
    return prompt

def build_llm_prompt_for_cluster_explanations(clustered_df):
    """
    Builds a prompt asking an LLM to explain clusters of features and propose
    possible business drivers.

    Parameters:
    - clustered_df: DataFrame with 'Feature' and 'Cluster' columns

    Returns:
    - prompt: String prompt for LLM
    """
    clusters = clustered_df.groupby("Cluster")["Feature"].apply(list)
    prompt = (
        "You are a risk model explanation assistant. Given the following clusters of features, "
        "please explain why the features in each cluster might share similar risk signals "
        "and suggest possible underlying business drivers:\n\n"
    )
    for cluster_id, features in clusters.items():
        feature_list = ", ".join(features)
        prompt += f"- Cluster {cluster_id}: {feature_list}\n"
    return prompt

def build_llm_prompt_for_action_suggestions(top_features, mean_abs_shap):
    """
    Builds a prompt asking an LLM to recommend business actions based on feature
    importance.

    Parameters:
    - top_features: List of top feature names
    - mean_abs_shap: Series of mean absolute SHAP values

    Returns:
    - prompt: String prompt for LLM
    """
    prompt = (
        "You are a risk model explanation assistant. Here are the top features contributing "
        "to high risk scores, along with their mean SHAP values:\n\n"
    )
    for feature in top_features:
        shap_value = round(mean_abs_shap[feature], 4)
        prompt += f"- {feature}: {shap_value}\n"

    prompt += (
        "\nFor each feature, recommend:\n"
        "- Investigative focus areas\n"
        "- Data quality checks\n"
        "- Policy or process improvements\n"
    )
    return prompt
