# def build_prompt(risk_score, top_features, feature_descriptions):
#     prompt = f"""
# You are a risk model explanation assistant. Given a risk score and a list of features with their descriptions and contributions, generate a clear, concise narrative explaining the risk score.

# Risk Score: {round(risk_score * 100, 2)}%

# Top Features and Contributions:
# """
#     total_abs = top_features["Abs SHAP Value"].sum()
#     for _, row in top_features.iterrows():
#         feature = row["Feature"]
#         contribution = round(row["Abs SHAP Value"] / total_abs * 100, 1)
#         description = feature_descriptions.get(feature, "No description available")
#         prompt += f"- {feature} ({description}): {contribution}% contribution\n"
#     prompt += "\nPlease produce a narrative that:\n- Starts with the risk score\n- Explains each feature’s contribution in plain language\n- Highlights why each feature might indicate a higher risk.\n"
#     return prompt

def build_prompt(risk_score, top_features, feature_descriptions):
    """
    Builds a prompt for a risk model explanation using SHAP values.

    Parameters:
    - risk_score (float): The risk score (probability between 0 and 1).
    - top_features (pd.DataFrame): Contains columns 'Feature', 'SHAP Value', 'Abs SHAP Value'.
    - feature_descriptions (dict): Mapping from feature name to business-friendly description.

    Returns:
    - str: A prompt to be used with an LLM.
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
        "- Explains how each feature contributed to the score, including whether it increased or decreased the risk.\n"
        "- Highlights why each feature might indicate higher or lower risk, using plain language for a business audience.\n"
    )

    return prompt

