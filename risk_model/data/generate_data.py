import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap


# Step 1: Create AML Feature Library
def create_realistic_aml_feature_library():
    features = [
        ("wirein_ct", "Number of wire inbound transactions"),
        ("wireout_ct", "Number of wire outbound transactions"),
        ("wirein_amt", "Total inbound wire amount"),
        ("wireout_amt", "Total outbound wire amount"),
        ("avg_txn_amt", "Average transaction amount"),
        ("high_risk_country_txn_pct", "Percentage of transactions with high-risk countries"),
        ("acct_age_days", "Account age in days"),
        ("num_sar_reports", "Number of SAR reports filed"),
        ("login_freq_30d", "Login frequency in the past 30 days"),
        ("geo_diversity_score", "Number of unique countries accessed from"),
    ]
    return pd.DataFrame(features, columns=["feature_name", "description"])


def load_aml_feature_library_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads AML feature library from a CSV file.

    Parameters:
    - csv_path (str): Path to the CSV file containing feature_name and description columns.

    Returns:
    - pd.DataFrame: DataFrame with columns ['feature_name', 'description']
    """
    feature_library_df = pd.read_csv(csv_path)

    # Optional: Validate the required columns exist
    required_cols = {"feature_name", "description"}
    if not required_cols.issubset(feature_library_df.columns):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    return feature_library_df


# Step 2: Simulate Feature Values by Feature Name
def simulate_feature_value_by_feature(feature_name):
    if "ct" in feature_name or "num" in feature_name or "freq" in feature_name:
        return int(np.random.poisson(lam=10))
    elif "amt" in feature_name:
        return round(np.random.uniform(1000, 100000), 2)
    elif "pct" in feature_name:
        return round(np.random.uniform(0, 1), 2)
    elif "score" in feature_name:
        return round(np.random.uniform(0, 10), 2)
    elif "age" in feature_name:
        return int(np.random.randint(30, 2000))
    else:
        return round(np.random.random(), 2)

# Step 3: Compute Contribution Percentages
def compute_contribution_percentages(shap_values):
    abs_vals = np.abs(shap_values)
    total_abs = abs_vals.sum()
    if total_abs == 0:
        return np.zeros_like(shap_values)
    return abs_vals / total_abs * 100

# Step 4: Build Structured Dataset with SHAP Values
def generate_structured_shap_dataset(feature_library, n_entities=100, base_value=0.2, output_path="structured_aml_dataset.json"):
    features = feature_library["feature_name"].tolist()
    num_features = len(features)
    data = []

    for entity_id in range(1, n_entities + 1):
        # Random SHAP values and scaling
        raw_shap = np.random.randn(num_features)
        shap_sum = raw_shap.sum()
        correction = ((np.random.rand() * 0.6) - 0.3) / shap_sum if shap_sum != 0 else 0
        shap_values = raw_shap * correction

        # Compute risk score and contributions
        risk_score = base_value + shap_values.sum()
        risk_score = float(np.clip(risk_score, 0, 1))
        contribution_pct = compute_contribution_percentages(shap_values)

        # Build feature dict
        feature_dict = {}
        for i, f in enumerate(features):
            feature_dict[f] = {
                "feature_value": simulate_feature_value_by_feature(f),
                "shap_value": float(shap_values[i]),
                "abs_shap_value": float(abs(shap_values[i])),
                "contribution_pct": float(contribution_pct[i])
            }

        data.append({
            "entity_id": entity_id,
            "risk_score": risk_score,
            "features": feature_dict
        })

    # Save to JSONL
    df = pd.DataFrame(data)
    df.to_json(output_path, orient="records", lines=True)
    print(f"✅ Structured dataset saved to {output_path}")
    return df


def load_shap_explanation(json_path, entity_id):
    with open(json_path, "r") as f:
        for line in f:
            entity = json.loads(line)
            if entity["entity_id"] == entity_id:
                break
        else:
            raise ValueError(f"Entity {entity_id} not found.")

    feature_names = list(entity["features"].keys())
    shap_vals = np.array([entity["features"][f]["shap_value"] for f in feature_names])
    data_vals = np.array([entity["features"][f]["feature_value"] for f in feature_names])

    explanation = shap.Explanation(
        values=shap_vals,
        data=data_vals,
        feature_names=feature_names,
        base_values=0.2  # use same base risk score as in simulation
    )
    return explanation


def build_feature_contribution_prompt_from_structured_json(
    entity_data: dict,
    feature_library_df: pd.DataFrame,
    top_n: int = 10
) -> str:
    """
    Builds a risk explanation prompt using structured SHAP data for a single entity.

    Parameters:
    - entity_data (dict): One JSON record from structured_aml_dataset.json, containing:
        'entity_id', 'risk_score', 'features': {feature_name: {feature_value, shap_value, contribution_pct, ...}}
    - feature_library_df (pd.DataFrame): DataFrame with:
        'feature_name', 'description'
    - top_n (int): Number of top features to include based on contribution_pct

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

    # Merge with feature library descriptions
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


def build_prompt_from_entity_row(entity_data: dict, feature_library_df: pd.DataFrame, top_n: int = 10) -> str:
    """
    Builds a prompt string from a single entity's data dictionary.

    Parameters:
    - entity_data (dict): Contains 'entity_id', 'risk_score', and 'features'
    - feature_library_df (pd.DataFrame): Feature descriptions with 'feature_name' and 'description'
    - top_n (int): Number of top features to include based on contribution percentage

    Returns:
    - str: Prompt ready to send to an LLM
    """
    entity_id = entity_data["entity_id"]
    risk_score = entity_data["risk_score"]
    features_dict = entity_data["features"]

    feature_rows = [
        {"feature_name": feat, "contribution_pct": vals["contribution_pct"]}
        for feat, vals in features_dict.items()
    ]
    feature_score_df = pd.DataFrame(feature_rows)

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


# def add_prompt_column(
#     entity_data_list: list,
#     feature_library_df: pd.DataFrame,
#     top_n: int = 10
# ) -> pd.DataFrame:
#     """
#     Expands entity data into a DataFrame of feature-level details and adds a prompt per entity.

#     Parameters:
#     - entity_data_list (list of dict): Each item contains 'entity_id', 'risk_score', and 'features'
#     - feature_library_df (pd.DataFrame): Feature descriptions
#     - top_n (int): Number of top features to include in the prompt

#     Returns:
#     - pd.DataFrame: One row per feature per entity, with prompt and all details
#     """
#     processed = []

#     for entity_data in entity_data_list:
#         entity_id = entity_data["entity_id"]
#         risk_score = entity_data["risk_score"]
#         features_dict = entity_data["features"]
#         prompt = build_prompt_from_entity_row(entity_data, feature_library_df, top_n)

#         for feat, vals in features_dict.items():
#             processed.append({
#                 "entity_id": entity_id,
#                 "feature_name": feat,
#                 "feature_value": vals.get("feature_value"),
#                 "shap_value": vals.get("shap_value"),
#                 "abs_shap_value": vals.get("abs_shap_value"),
#                 "contribution_pct": vals.get("contribution_pct"),
#                 "risk_score": risk_score,
#                 "prompt": prompt
#             })

#     return pd.DataFrame(processed)

def add_prompt_to_entity_json(
    entity_data_list: list,
    feature_library_df: pd.DataFrame,
    top_n: int = 10
) -> list:
    """
    Adds a 'prompt' field to each entity dictionary based on its top features.

    Parameters:
    - entity_data_list (list of dict): Each entity dict includes 'entity_id', 'risk_score', and 'features'
    - feature_library_df (pd.DataFrame): Feature descriptions
    - top_n (int): Number of top features to include in the prompt

    Returns:
    - list of dict: Same format as input but with added 'prompt' field
    """
    updated_entities = []

    for entity_data in entity_data_list:
        # Generate explanation prompt
        prompt = build_prompt_from_entity_row(entity_data, feature_library_df, top_n)
        
        # Add prompt to entity dict
        entity_with_prompt = entity_data.copy()
        entity_with_prompt["prompt"] = prompt

        updated_entities.append(entity_with_prompt)

    return updated_entities


def save_entities_to_json(entities: list, file_path: str):
    """
    Save the list of entities (dicts) to a JSON file with indentation.

    Parameters:
    - entities (list of dict): Entities to save
    - file_path (str): Output JSON file path
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    print(f"✅ Entities saved to {file_path}")


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

# Import or define build_prompt_from_entity_row here
# from your_module import build_prompt_from_entity_row
    
def enrich_entities_with_llm_explanations(
    input_json_path: str,
    output_json_path: str,
    feature_library_df: pd.DataFrame,
    azure_client: AzureClient,
    top_n: int = 10,
    log_every_n: int = 5,  # New parameter for logging frequency
):
    """
    Enrich entities with LLM-generated explanations and save the updated list to a JSON file.

    Parameters:
    - input_json_path (str): Path to input JSON file with entity data.
    - output_json_path (str): Path to save enriched JSON file.
    - feature_library_df (pd.DataFrame): DataFrame with 'feature_name' and 'description'.
    - azure_client (AzureClient): Initialized Azure OpenAI client.
    - top_n (int): Number of top features to use in prompt generation.
    - log_every_n (int): Frequency of logging progress (e.g., 1 = every entity, 10 = every 10th entity).
    """
    # Load existing entities JSON file
    with open(input_json_path, "r") as f:
        entity_data_list = json.load(f)

    updated_entities = []

    for i, entity_data in enumerate(entity_data_list):
        prompt = build_prompt_from_entity_row(entity_data, feature_library_df, top_n)
        explanation = azure_client.get_response(prompt)
        entity_data["prompt"] = prompt
        entity_data["llm_explanation"] = explanation
        updated_entities.append(entity_data)

        if (i + 1) % log_every_n == 0 or i == len(entity_data_list) - 1:
            print(f"Processed entity {i + 1}/{len(entity_data_list)}")

    # Save updated entities back to JSON file
    with open(output_json_path, "w") as f:
        json.dump(updated_entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved enriched entities with explanations to {output_json_path}")


def enrich_and_evaluate_entities(
    input_json_path: str,
    output_json_path: str,
    feature_library_df: pd.DataFrame,
    azure_client: AzureClient,
    judge_models: dict = JUDGE_MODELS,
    top_n: int = 10,
    log_every_n: int = 5
):
    # Load input JSON
    with open(input_json_path, "r", encoding="utf-8") as f:
        entity_data_list = json.load(f)

    enriched_entities = []

    for i, entity in enumerate(entity_data_list):
        # Build prompt and get explanation
        prompt = build_prompt_from_entity_row(entity, feature_library_df, top_n)
        explanation = azure_client.get_response(prompt)

        # Attach to entity
        entity["prompt"] = prompt
        entity["llm_explanation"] = explanation

        # Evaluate using judge models
        entity["evaluations"] = {}
        for judge_name, deployment_name in judge_models.items():
            eval_prompt = build_evaluation_prompt(prompt, explanation)

            start_time = datetime.now()
            eval_response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates AI-generated text."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            end_time = datetime.now()

            eval_text = eval_response['choices'][0]['message']['content']
            clarity, conciseness, completeness = extract_scores(eval_text)
            eval_duration = (end_time - start_time).total_seconds()

            entity["evaluations"][judge_name] = {
                "Clarity": clarity,
                "Conciseness": conciseness,
                "Completeness": completeness,
                "Summary": eval_text.strip(),
                "EvalTime": eval_duration
            }

        enriched_entities.append(entity)

        if (i + 1) % log_every_n == 0 or i == len(entity_data_list) - 1:
            print(f"✅ Processed {i + 1}/{len(entity_data_list)} entities")

    # Save back to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Enriched and evaluated entities saved to {output_json_path}")



def update_mean_std_scores_in_json(input_json_path: str, output_json_path: str):
    """
    Load enriched entity JSON file, compute mean and standard deviation for all
    judge model evaluation scores, and save updated entities back to JSON.

    Parameters:
    - input_json_path: Path to input JSON file with judge model scores.
    - output_json_path: Path to output JSON file to save updated entities.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        entity_data_list = json.load(f)

    for entity in entity_data_list:
        # Collect scores by dimension
        clarity_scores = []
        conciseness_scores = []
        completeness_scores = []

        for key, value in entity.items():
            if key.startswith("Clarity_"):
                clarity_scores.append(value)
            elif key.startswith("Conciseness_"):
                conciseness_scores.append(value)
            elif key.startswith("Completeness_"):
                completeness_scores.append(value)

        # Compute mean and std
        def calc_stats(scores):
            return {
                "mean": float(np.mean(scores)) if scores else None,
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None
            }

        entity["Clarity_stats"] = calc_stats(clarity_scores)
        entity["Conciseness_stats"] = calc_stats(conciseness_scores)
        entity["Completeness_stats"] = calc_stats(completeness_scores)

    # Save updated JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entity_data_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated entities saved to: {output_json_path}")


# Step 5: Run the Full Pipeline
if __name__ == "__main__":
    feature_lib = create_realistic_aml_feature_library()
    df_structured = generate_structured_shap_dataset(feature_lib, n_entities=100, output_path="structured_aml_dataset.json")
    explanation = load_shap_explanation("structured_aml_dataset.json", entity_id=1)
    shap.plots.bar(explanation)
    
    # with open("structured_aml_dataset.json", "r") as f:
    #     entity_data = json.loads(f.readline())  # Get first entity
    
    # feature_library_df = create_realistic_aml_feature_library()
    # prompt = build_feature_contribution_prompt_from_structured_json(entity_data, feature_library_df, top_n=5)
    # print(prompt)


    # Load feature library
    feature_library_df = create_realistic_aml_feature_library()
    
    # Load structured dataset
    with open("structured_aml_dataset.json", "r") as f:
        entity_data_list = [json.loads(line) for line in f]
    
    # Add prompt column
    df_with_prompts = add_prompt_column(entity_data_list, feature_library_df, top_n=5)
    
    # Save or inspect
    # df_with_prompts.to_csv("aml_prompts.csv", index=False)
    # print(df_with_prompts.head(1)["prompt"].values[0])

    # # Setup your AzureClient instance
    # client = AzureClient()

    # # Load your feature library DataFrame from CSV or other source
    # feature_library_df = pd.read_csv("feature_library.csv")

    # # Paths
    # input_json = "entities_input.json"  # Your existing dataset file
    # output_json = "entities_with_llm_explanations.json"

    # enrich_entities_with_llm_explanations(
    #     input_json_path=input_json,
    #     output_json_path=output_json,
    #     feature_library_df=feature_library_df,
    #     azure_client=client,
    #     top_n=10,
    # )

    enrich_entities_with_llm_explanations(
        input_json_path="entities_with_prompts.json",
        output_json_path="entities_with_llm_explanations.json",
        feature_library_df=feature_library_df,
        azure_client=azure_client,
        top_n=5,
        log_every_n=5  # Print status every 5 entities
    )

    enrich_and_evaluate_entities(
        input_json_path="entities_with_prompts.json",
        output_json_path="entities_enriched_evaluated.json",
        feature_library_df=feature_library_df,
        azure_client=azure_client,
        top_n=5,
        log_every_n=5  # print progress every 3 entities
    )

    update_mean_std_scores_in_json(
        input_json_path="entities_with_evals.json",
        output_json_path="entities_with_stats.json"
    )