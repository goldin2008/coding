"""
prompt_generator.py

This module contains functions to build various prompts for LLMs:
- risk score explanation with SHAP values
- model output narrative generation
- evaluation prompts to judge narrative quality
"""

import pandas as pd


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


# def build_evaluation_prompt(generated_text: str) -> str:
#     """
#     Builds the prompt text for evaluating a risk explanation.

#     Parameters:
#     - generated_text (str): The risk explanation to evaluate.

#     Returns:
#     - str: The evaluation prompt text.
#     """
#     evaluation_prompt = f"""
#         Please evaluate the following risk explanation on the following criteria (scale 1-5):
#         1. Clarity
#         2. Conciseness
#         3. Completeness
        
#         Provide a short justification for each score.
        
#         Generated Explanation:
#         \"\"\"
#         {generated_text}
#         \"\"\"
#         """
#     return evaluation_prompt.strip()

def build_evaluation_prompt(prompt: str, explanation: str) -> str:
    """
    Construct the evaluation prompt given the original prompt and model-generated explanation.
    """
    return f"""
You are an evaluation assistant. Given a prompt and a model-generated answer, please assess the quality of the answer based on:
- Clarity (1-5)
- Conciseness (1-5)
- Completeness (1-5)
Provide a score for each, then write a short summary comment.

Prompt:
{prompt}

Model-Generated Answer:
{explanation}
"""



def build_judge_prompt(human_narrative, llm_narrative):
    """
    Builds a prompt for evaluating a human vs. LLM-generated narrative.

    Parameters:
    - human_narrative (str): Narrative written by a human.
    - llm_narrative (str): Narrative generated by the LLM.

    Returns:
    - str: Prompt to be sent to an LLM to evaluate the quality of both narratives.
    """
    prompt = (
        "You are an evaluation assistant. "
        "You will compare two narratives explaining a risk score for a financial model. "
        "The first narrative is from a human expert, and the second narrative is generated by a language model. "
        "Please rate the clarity, completeness, and overall quality of each narrative on a scale from 1 to 5, "
        "and then provide an overall judgment of which narrative is better.\n\n"
        f"Human Narrative:\n{human_narrative.strip()}\n\n"
        f"LLM-Generated Narrative:\n{llm_narrative.strip()}\n\n"
        "Please provide:\n"
        "1. A table comparing clarity, completeness, and quality for each narrative.\n"
        "2. An overall rating indicating which narrative is better and why.\n"
    )

    return prompt


def build_single_evaluation_prompt(prompt_text, generated_text):
    """
    Builds a prompt for evaluating a single model-generated narrative.

    Parameters:
    - prompt_text (str): The prompt that was originally sent to the LLM.
    - generated_text (str): The narrative generated by the LLM.

    Returns:
    - str: Prompt to be sent to an LLM to evaluate the quality of the generated text.
    """
    prompt = (
        "You are an evaluation assistant. "
        "Given a prompt and a model-generated answer, please assess the quality of the answer based on:\n"
        "- Clarity (1-5)\n"
        "- Conciseness (1-5)\n"
        "- Completeness (1-5)\n"
        "Provide a score for each, then write a short summary comment.\n\n"
        f"Prompt:\n{prompt_text.strip()}\n\n"
        f"Model-Generated Answer:\n{generated_text.strip()}\n"
    )
    return prompt