"""
prompt_generator.py

This module contains functions to build various prompts for LLMs:
- risk score explanation with SHAP values
- model output narrative generation
- evaluation prompts to judge narrative quality
"""

import pandas as pd
from typing import Dict, List, Optional


"""
This module provides a function to prepare a risk model explanation prompt
based on sorted feature scores, feature descriptions, and risk score.
"""
def build_feature_contribution_prompt(
    feature_score_df: pd.DataFrame,
    feature_library_df: pd.DataFrame,
    risk_score_df: pd.DataFrame
) -> str:
    """
    Builds a risk explanation prompt using feature contributions.

    Parameters:
    - feature_score_df (pd.DataFrame): DataFrame with columns:
        'feature_name', 'score'.
    - feature_library_df (pd.DataFrame): DataFrame with columns:
        'feature_name', 'feature_meaning'.
    - risk_score_df (pd.DataFrame): DataFrame with at least one row and column:
        'risk_score'.

    Returns:
    - str: Prompt text ready to be sent to the LLM.
    """
    # Sort feature scores by descending contribution
    feature_score_df = feature_score_df.sort_values(by="score", ascending=False)

    # Merge the scores with the feature meanings
    merged_df = pd.merge(
        feature_score_df,
        feature_library_df,
        on="feature_name",
        how="left"
    )

    # Extract the risk score (assuming first row)
    risk_score = risk_score_df.iloc[0]["risk_score"]

    # Format features for the prompt
    features_text = ""
    for _, row in merged_df.iterrows():
        feature_name = row["feature_name"]
        feature_meaning = row.get("feature_meaning", "No description available")
        score = row["score"]
        features_text += (
            f"- {feature_name} ({feature_meaning}): {score:.0%} contribution\n"
        )

    # Build the prompt
    prompt_text = (
        "You are a risk model explanation assistant. "
        "Given a risk score and a list of features with their descriptions and contributions, "
        "generate a clear, concise narrative explaining the risk score.\n\n"
        f"Risk Score: {risk_score:.0%}\n"
        "Top Features and Contributions:\n"
        f"{features_text}\n"
        "Please produce a narrative that:\n"
        "- Starts with the risk score\n"
        "- Explains each feature’s contribution in plain language\n"
        "- Highlights why each feature might indicate a higher risk\n"
    )

    return prompt_text


def build_shap_explanation_prompt(risk_score, top_features, feature_descriptions):
    """
    Builds a prompt for a risk model explanation using SHAP values.

    Parameters:
    - risk_score (float): The risk score (probability between 0 and 1).
    - top_features (pd.DataFrame): DataFrame with columns 'Feature', 'SHAP Value', 'Abs SHAP Value'.
    - feature_descriptions (dict): Mapping from feature name to business-friendly description.

    Returns:
    - str: Prompt to be sent to the LLM.
    """
    prompt = (
        "You are a risk model explanation assistant. "
        "Given a risk score and a list of features with their descriptions and contributions, "
        "generate a clear, concise narrative explaining the risk score.\n\n"
        f"Risk Score: {risk_score * 100:.2f}%\n\n"
        "Top Features and Contributions:\n"
    )

    total_abs = top_features["Abs SHAP Value"].sum()

    for _, row in top_features.iterrows():
        feature = row["Feature"]
        shap_value = row["SHAP Value"]
        direction = "increased" if shap_value > 0 else "decreased"
        contribution = round(row["Abs SHAP Value"] / total_abs * 100, 1)
        description = feature_descriptions.get(feature, "No description available")
        prompt += (
            f"- {feature} ({description}): This feature {direction} the risk score, "
            f"contributing {contribution}% of the total impact.\n"
        )

    prompt += (
        "\nPlease produce a narrative that:\n"
        "- Starts with the risk score.\n"
        "- Explains how each feature contributed to the score (including whether it increased or decreased the risk).\n"
        "- Uses plain language suitable for a business audience.\n"
    )

    return prompt







# def build_feature_contribution_prompt_from_structured_json(
#     entity_data: dict,
#     feature_library_df: pd.DataFrame,
#     top_n: int = 10
# ) -> str:
#     """
#     Builds a risk explanation prompt using structured SHAP data for a single entity.

#     Parameters:
#     - entity_data (dict): One JSON record from structured_aml_dataset.json, containing:
#         'entity_id', 'risk_score', 'features': {feature_name: {feature_value, shap_value, contribution_pct, ...}}
#     - feature_library_df (pd.DataFrame): DataFrame with:
#         'feature_name', 'description'
#     - top_n (int): Number of top features to include based on contribution_pct

#     Returns:
#     - str: Prompt ready to send to an LLM.
#     """
#     entity_id = entity_data["entity_id"]
#     risk_score = entity_data["risk_score"]
#     features_dict = entity_data["features"]

#     # Convert nested features to DataFrame
#     feature_rows = [
#         {
#             "feature_name": feature,
#             "contribution_pct": values["contribution_pct"]
#         }
#         for feature, values in features_dict.items()
#     ]
#     feature_score_df = pd.DataFrame(feature_rows)

#     # Merge with feature library descriptions
#     merged_df = pd.merge(
#         feature_score_df,
#         feature_library_df.rename(columns={"feature_name": "feature_name", "description": "feature_meaning"}),
#         on="feature_name",
#         how="left"
#     ).sort_values(by="contribution_pct", ascending=False).head(top_n)

#     # Build feature list string
#     features_text = ""
#     for _, row in merged_df.iterrows():
#         features_text += (
#             f"- {row['feature_name']} ({row.get('feature_meaning', 'No description available')}): "
#             f"{row['contribution_pct']:.1f}% contribution\n"
#         )

#     # Final prompt
#     prompt_text = (
#         f"You are a risk model explanation assistant. "
#         f"Given a risk score and a list of features with their descriptions and contributions, "
#         f"generate a clear, concise narrative explaining the risk score for entity ID {entity_id}.\n\n"
#         f"Entity ID: {entity_id}\n"
#         f"Risk Score: {risk_score:.0%}\n"
#         "Top Features and Contributions:\n"
#         f"{features_text}\n"
#         "Please produce a narrative that:\n"
#         "- Starts with the risk score\n"
#         "- Explains each feature’s contribution in plain language\n"
#         "- Highlights why each feature might indicate a higher risk\n"
#     )
#     return prompt_text
def build_feature_contribution_prompt_from_structured_json(
    entity_data: dict,
    feature_library_df: pd.DataFrame,
    top_n: int = 10,
    selected_features: list[str] = None
) -> str:
    """
    Builds a risk explanation prompt using structured SHAP data for a single entity.

    Parameters:
    - entity_data (dict): One JSON record from structured_aml_dataset.json, containing:
        'entity_id', 'risk_score', 'features': {feature_name: {feature_value, shap_value, contribution_pct, ...}}
    - feature_library_df (pd.DataFrame): DataFrame with:
        'feature_name', 'description'
    - top_n (int): Number of top features to include based on contribution_pct
    - selected_features (list[str], optional): If provided, only consider these features before ranking

    Returns:
    - str: Prompt ready to send to an LLM.
    """
    entity_id = entity_data["entity_id"]
    risk_score = entity_data["risk_score"]
    features_dict = entity_data["features"]

    # Convert nested features to DataFrame
    feature_rows = [
        {
            "feature_name": feature,
            "contribution_pct": values["contribution_pct"]
        }
        for feature, values in features_dict.items()
    ]
    feature_score_df = pd.DataFrame(feature_rows)

    # Optional filtering by selected feature names
    if selected_features is not None:
        feature_score_df = feature_score_df[feature_score_df["feature_name"].isin(selected_features)]

    # Merge with descriptions
    merged_df = pd.merge(
        feature_score_df,
        feature_library_df.rename(columns={"feature_name": "feature_name", "description": "feature_meaning"}),
        on="feature_name",
        how="left"
    ).sort_values(by="contribution_pct", ascending=False).head(top_n)

    # Build feature list string
    features_text = ""
    for _, row in merged_df.iterrows():
        features_text += (
            f"- {row['feature_name']} ({row.get('feature_meaning', 'No description available')}): "
            f"{row['contribution_pct']:.1f}% contribution\n"
        )

    # Final prompt
    prompt_text = (
        f"You are a risk model explanation assistant. "
        f"Given a risk score and a list of features with their descriptions and contributions, "
        f"generate a clear, concise narrative explaining the risk score for entity ID {entity_id}.\n\n"
        f"Entity ID: {entity_id}\n"
        f"Risk Score: {risk_score:.0%}\n"
        "Top Features and Contributions:\n"
        f"{features_text}\n"
        "Please produce a narrative that:\n"
        "- Starts with the risk score\n"
        "- Explains each feature’s contribution in plain language\n"
        "- Highlights why each feature might indicate a higher risk\n"
    )
    return prompt_text


# def build_prompt_from_entity_row(entity_data: dict, feature_library_df: pd.DataFrame, top_n: int = 10) -> str:
#     """
#     Builds a prompt string from a single entity's data dictionary.

#     Parameters:
#     - entity_data (dict): Contains 'entity_id', 'risk_score', and 'features'
#     - feature_library_df (pd.DataFrame): Feature descriptions with 'feature_name' and 'description'
#     - top_n (int): Number of top features to include based on contribution percentage

#     Returns:
#     - str: Prompt ready to send to an LLM
#     """
#     entity_id = entity_data["entity_id"]
#     risk_score = entity_data["risk_score"]
#     features_dict = entity_data["features"]

#     feature_rows = [
#         {"feature_name": feat, "contribution_pct": vals["contribution_pct"]}
#         for feat, vals in features_dict.items()
#     ]
#     feature_score_df = pd.DataFrame(feature_rows)

#     merged_df = pd.merge(
#         feature_score_df,
#         feature_library_df.rename(columns={"feature_name": "feature_name", "description": "feature_meaning"}),
#         on="feature_name",
#         how="left"
#     ).sort_values(by="contribution_pct", ascending=False).head(top_n)

#     features_text = ""
#     for _, row in merged_df.iterrows():
#         features_text += (
#             f"- {row['feature_name']} ({row.get('feature_meaning', 'No description available')}): "
#             f"{row['contribution_pct']:.1f}% contribution\n"
#         )

#     prompt = (
#         f"You are a risk model explanation assistant. "
#         f"Given a risk score and a list of features with their descriptions and contributions, "
#         f"generate a clear, concise narrative explaining the risk score for entity ID {entity_id}.\n\n"
#         f"Entity ID: {entity_id}\n"
#         f"Risk Score: {risk_score:.0%}\n"
#         f"Top Features and Contributions:\n{features_text}\n"
#         f"Please produce a narrative that:\n"
#         f"- Starts with the risk score\n"
#         f"- Explains each feature’s contribution in plain language\n"
#         f"- Highlights why each feature might indicate a higher risk\n"
#     )
#     return prompt

def build_prompt_from_entity_row(entity_data: dict, feature_library_df: pd.DataFrame, top_n: int = 10, selected_features: list[str] = None) -> str:
    """
    Builds a prompt string from a single entity's data dictionary.

    Parameters:
    - entity_data (dict): Contains 'entity_id', 'risk_score', and 'features'
    - feature_library_df (pd.DataFrame): Feature descriptions with 'feature_name' and 'description'
    - top_n (int): Number of top features to include based on contribution percentage
    - selected_features (list[str], optional): List of feature names to include before ranking

    Returns:
    - str: Prompt ready to send to an LLM
    """
    entity_id = entity_data["entity_id"]
    risk_score = entity_data["risk_score"]
    features_dict = entity_data["features"]

    # Step 1: Convert features to DataFrame
    feature_rows = [
        {"feature_name": feat, "contribution_pct": vals["contribution_pct"]}
        for feat, vals in features_dict.items()
    ]
    feature_score_df = pd.DataFrame(feature_rows)

    # Step 2: Optional filtering by selected features
    if selected_features is not None:
        feature_score_df = feature_score_df[feature_score_df["feature_name"].isin(selected_features)]

    # Step 3: Merge with feature descriptions
    merged_df = pd.merge(
        feature_score_df,
        feature_library_df.rename(columns={"feature_name": "feature_name", "description": "feature_meaning"}),
        on="feature_name",
        how="left"
    )

    # Step 4: Rank and pick top N by contribution
    top_features_df = merged_df.sort_values(by="contribution_pct", ascending=False).head(top_n)

    # Step 5: Format prompt body
    features_text = ""
    for _, row in top_features_df.iterrows():
        features_text += (
            f"- {row['feature_name']} ({row.get('feature_meaning', 'No description available')}): "
            f"{row['contribution_pct']:.1f}% contribution\n"
        )

    # Step 6: Build final prompt
    prompt = (
        f"You are a risk model explanation assistant. "
        f"Given a risk score and a list of features with their descriptions and contributions, "
        f"generate a clear, concise narrative explaining the risk score for entity ID {entity_id}.\n\n"
        f"Entity ID: {entity_id}\n"
        f"Risk Score: {risk_score:.0%}\n"
        f"Top Features and Contributions:\n{features_text}\n"
        f"Please produce a narrative that:\n"
        f"- Starts with the risk score\n"
        f"- Explains each feature’s contribution in plain language\n"
        f"- Highlights why each feature might indicate a higher risk\n"
    )
    return prompt


def build_controlled_prompt_from_entity_row(
    entity_data: dict,
    feature_library_df: pd.DataFrame,
    ccc_levels: dict = {"clarity": 3, "conciseness": 3, "completeness": 3},
    top_n: int = 10
) -> str:
    """
    Builds a prompt from entity data with CCC quality level instructions.

    Parameters:
    - entity_data (dict): Must include 'entity_id', 'risk_score', and 'features'
    - feature_library_df (pd.DataFrame): Includes 'feature_name' and 'description'
    - ccc_levels (dict): Quality control levels, e.g., {"clarity": 2, "conciseness": 3, "completeness": 1}
    - top_n (int): Number of top features to include

    Returns:
    - str: Prompt for LLM that includes CCC requirements
    """
    # Unpack CCC levels with defaults
    clarity = ccc_levels.get("clarity", 3)
    conciseness = ccc_levels.get("conciseness", 3)
    completeness = ccc_levels.get("completeness", 3)

    entity_id = entity_data["entity_id"]
    risk_score = entity_data["risk_score"]
    features_dict = entity_data["features"]

    # Create feature score DataFrame
    feature_rows = [
        {"feature_name": feat, "contribution_pct": vals["contribution_pct"]}
        for feat, vals in features_dict.items()
    ]
    feature_score_df = pd.DataFrame(feature_rows)

    # Join with descriptions
    merged_df = pd.merge(
        feature_score_df,
        feature_library_df.rename(columns={"feature_name": "feature_name", "description": "feature_meaning"}),
        on="feature_name",
        how="left"
    ).sort_values(by="contribution_pct", ascending=False).head(top_n)

    features_text = ""
    for _, row in merged_df.iterrows():
        features_text += (
            f"- {row['feature_name']} ({row.get('feature_meaning', 'No description available')}): "
            f"{row['contribution_pct']:.1f}% contribution\n"
        )

    prompt = (
        f"You are a risk model explanation assistant.\n"
        f"Your task is to generate a narrative explanation of a risk score using features and their contributions.\n\n"
        f"⚠️ Please follow these quality targets in your explanation:\n"
        f"- Clarity Level: {clarity} (1=low, 3=high)\n"
        f"- Conciseness Level: {conciseness} (1=verbose, 3=succinct)\n"
        f"- Completeness Level: {completeness} (1=partial, 3=fully explained)\n\n"
        f"Entity ID: {entity_id}\n"
        f"Risk Score: {risk_score:.0%}\n"
        f"Top Features and Contributions:\n{features_text}\n"
        f"Instructions:\n"
        f"- Start with the risk score\n"
        f"- Explain each feature’s contribution in plain language\n"
        f"- Highlight why each feature indicates risk\n"
        f"- Follow the CCC levels provided\n"
    )

    return prompt