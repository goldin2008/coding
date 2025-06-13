import re
from datetime import datetime
import pandas as pd
import openai  # Ensure your openai client is configured before use

# Judge model deployment names (replace with your actual Azure OpenAI deployment names)
JUDGE_MODELS = {
    "GPT-4": "gpt-4-deployment",
    "GPT-4o": "gpt-4o-deployment",
    "GPT-35-Turbo": "gpt-35-turbo-deployment"
}

def build_evaluation_prompt(prompt: str, output: str) -> str:
    """
    Construct the evaluation prompt given the original prompt and model-generated output.
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
{output}
"""


def evaluate_with_judge(azure_client, deployment_name: str, prompt: str, output: str) -> str:
    """
    Send evaluation request to Azure OpenAI judge model via AzureClient and return the evaluation text.

    Parameters:
    - azure_client: Your AzureClient instance
    - deployment_name (str): The name of the deployment (judge model)
    - prompt (str): The original explanation prompt
    - output (str): The model's explanation output

    Returns:
    - str: Judge model's evaluation response
    """
    eval_prompt = build_evaluation_prompt(prompt, output)
    return azure_client.get_response(eval_prompt, deployment_name=deployment_name)


def extract_scores(eval_text: str):
    """
    Extract clarity, conciseness, and completeness scores from the evaluation text.
    Returns a tuple of (clarity, conciseness, completeness), each as int or None if not found.
    """
    clarity = conciseness = completeness = None

    clarity_match = re.search(r'Clarity[:\s]+(\d)', eval_text, re.IGNORECASE)
    conciseness_match = re.search(r'Conciseness[:\s]+(\d)', eval_text, re.IGNORECASE)
    completeness_match = re.search(r'Completeness[:\s]+(\d)', eval_text, re.IGNORECASE)

    if clarity_match:
        clarity = int(clarity_match.group(1))
    if conciseness_match:
        conciseness = int(conciseness_match.group(1))
    if completeness_match:
        completeness = int(completeness_match.group(1))

    return clarity, conciseness, completeness

def evaluate_dataframe(df: pd.DataFrame, prompt_col: str, output_col: str, 
                       judge_models: dict = JUDGE_MODELS) -> pd.DataFrame:
    """
    Evaluate a DataFrame with prompt and output columns using multiple judge models.
    Returns a new DataFrame with evaluation scores and summaries for each row and judge model.

    Output columns added per judge model (e.g., for GPT-4):
      - Clarity_GPT-4
      - Conciseness_GPT-4
      - Completeness_GPT-4
      - EvalSummary_GPT-4
      - EvalTime_GPT-4

    Parameters:
    - df: input DataFrame
    - prompt_col: name of the column containing the prompt text
    - output_col: name of the column containing the generated answer text
    - judge_models: dict of judge model names and deployment names (default JUDGE_MODELS)
    """
    df = df.copy()
    for judge_name, deployment_name in judge_models.items():
        clarity_list = []
        conciseness_list = []
        completeness_list = []
        eval_summary_list = []
        eval_time_list = []

        print(f"Starting evaluation with judge model: {judge_name}")

        for idx, row in df.iterrows():
            prompt_text = row[prompt_col]
            output_text = row[output_col]

            start_time = datetime.now()
            eval_text = evaluate_with_judge(deployment_name, prompt_text, output_text)
            end_time = datetime.now()

            clarity, conciseness, completeness = extract_scores(eval_text)
            eval_duration = (end_time - start_time).total_seconds()

            clarity_list.append(clarity)
            conciseness_list.append(conciseness)
            completeness_list.append(completeness)
            eval_summary_list.append(eval_text.strip())
            eval_time_list.append(eval_duration)

        # Add columns to DataFrame
        df[f"Clarity_{judge_name}"] = clarity_list
        df[f"Conciseness_{judge_name}"] = conciseness_list
        df[f"Completeness_{judge_name}"] = completeness_list
        df[f"EvalSummary_{judge_name}"] = eval_summary_list
        df[f"EvalTime_{judge_name}"] = eval_time_list

    return df
