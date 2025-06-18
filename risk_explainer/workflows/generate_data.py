import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap
from clients.azure_openai_client import AzureClient
from evaluation.judge import JUDGE_MODELS
import re
from datetime import datetime
from evaluation.judge import *
from prompts.prompt_generator import *

from typing import List, Optional, Dict, Any

from tqdm import tqdm

from matplotlib.patches import Patch

from scipy.interpolate import make_interp_spline


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


def load_aml_feature_library_from_excel(csv_path: str) -> pd.DataFrame:
    """
    Loads AML feature library from a CSV file.

    Parameters:
    - csv_path (str): Path to the CSV file containing feature_name and description columns.

    Returns:
    - pd.DataFrame: DataFrame with columns ['feature_name', 'description']
    """
    feature_library_df = pd.read_excel(csv_path)

    # Optional: Validate the required columns exist
    required_cols = {"Feature Name", "llm_explanation"}
    if not required_cols.issubset(feature_library_df.columns):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

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

def generate_mock_judges(
    num_judges: int = 3,
    name_prefix: str = "MockJudge",
    deployment_prefix: str = "mock_judge_model",
    start_index: int = 1
) -> dict:
    """
    Generates mock judge models with flexible naming
    
    Args:
        num_judges: Number of judges to generate
        name_prefix: Prefix for display names
        deployment_prefix: Prefix for deployment names
        start_index: Starting index number
        
    Returns:
        Dictionary of {display_name: deployment_name}
    """
    return {
        f"{name_prefix}{i}": f"{deployment_prefix}{i}"
        for i in range(start_index, start_index + num_judges)
    }


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


# Import or define build_prompt_from_entity_row here
# from your_module import build_prompt_from_entity_row

# def add_prompt_to_entity_json(
#     entity_data_list: list,
#     feature_library_df: pd.DataFrame,
#     top_n: int = 10
# ) -> list:
#     """
#     Adds a 'prompt' field to each entity dictionary based on its top features.

#     Parameters:
#     - entity_data_list (list of dict): Each entity dict includes 'entity_id', 'risk_score', and 'features'
#     - feature_library_df (pd.DataFrame): Feature descriptions
#     - top_n (int): Number of top features to include in the prompt

#     Returns:
#     - list of dict: Same format as input but with added 'prompt' field
#     """
#     updated_entities = []

#     for entity_data in entity_data_list:
#         # Generate explanation prompt
#         prompt = build_prompt_from_entity_row(entity_data, feature_library_df, top_n)
        
#         # Add prompt to entity dict
#         entity_with_prompt = entity_data.copy()
#         entity_with_prompt["prompt"] = prompt

#         updated_entities.append(entity_with_prompt)

#     return updated_entities
def add_prompt_to_entity_json(
    entity_data_list: list,
    feature_library_df: pd.DataFrame,
    top_n: int = 10,
    selected_features: list[str] = None
) -> list:
    """
    Adds a 'prompt' field to each entity dictionary based on its top features.

    Parameters:
    - entity_data_list (list of dict): Each entity dict includes 'entity_id', 'risk_score', and 'features'
    - feature_library_df (pd.DataFrame): Feature descriptions
    - top_n (int): Number of top features to include in the prompt
    - selected_features (list of str, optional): If provided, only use these features before top-n selection

    Returns:
    - list of dict: Same format as input but with added 'prompt' field
    """
    updated_entities = []

    for entity_data in entity_data_list:
        # Generate explanation prompt with optional filtering
        prompt = build_prompt_from_entity_row(
            entity_data=entity_data,
            feature_library_df=feature_library_df,
            top_n=top_n,
            selected_features=selected_features
        )

        # Add prompt to entity dict
        entity_with_prompt = entity_data.copy()
        entity_with_prompt["prompt"] = prompt

        updated_entities.append(entity_with_prompt)

    return updated_entities



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
        # prompt = build_prompt_from_entity_row(entity_data, feature_library_df, top_n)
        prompt = entity_data["prompt"]
        explanation = azure_client.get_response(prompt)
        # entity_data["prompt"] = prompt
        entity_data["llm_explanation"] = explanation
        updated_entities.append(entity_data)

        if (i + 1) % log_every_n == 0 or i == len(entity_data_list) - 1:
            print(f"Processed entity {i + 1}/{len(entity_data_list)}")

    # Save updated entities back to JSON file
    with open(output_json_path, "w") as f:
        json.dump(updated_entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved enriched entities with explanations to {output_json_path}")

    return updated_entities


def enrich_and_evaluate_entities(
    input_json_path: str,
    output_json_path: str,
    feature_library_df: pd.DataFrame,
    azure_client: AzureClient,
    judge_models: dict = JUDGE_MODELS,
    top_n: int = 10,
    log_every_n: int = 5
):
    """
    Evaluates pre-existing LLM explanations using judge models.

    Parameters:
    - input_json_path: Path to input JSON file (must include 'prompt' and 'llm_explanation')
    - output_json_path: Path to save enriched JSON with evaluation scores
    - feature_library_df: DataFrame with feature descriptions (optional if prompt is reused)
    - azure_client: AzureClient instance for calling judge model
    - judge_models: Dictionary of judge model names and deployment names
    - top_n: Number of top features to use in prompt (ignored since prompt is reused)
    - log_every_n: Frequency of logging progress
    """
    # Load input
    with open(input_json_path, "r", encoding="utf-8") as f:
        entity_data_list = json.load(f)

    enriched_entities = []

    for i, entity in enumerate(entity_data_list):
        prompt = entity.get("prompt", "")
        explanation = entity.get("llm_explanation", "")

        if not prompt or not explanation:
            print(f"⚠️ Entity {i+1} missing prompt or explanation. Skipping.")
            continue

        entity["evaluations"] = {}

        for judge_name, deployment_name in judge_models.items():
            eval_prompt = build_evaluation_prompt(prompt, explanation)

            start_time = datetime.now()
            eval_text = azure_client.get_response(eval_prompt, model_name=deployment_name)
            end_time = datetime.now()

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

    # Save updated entities
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluations completed and saved to {output_json_path}")

    return enriched_entities


# def update_mean_std_scores_in_json(input_json_path: str, output_json_path: str):
#     """
#     Load enriched entity JSON file, compute mean and standard deviation for all
#     judge model evaluation scores, and save updated entities back to JSON.

#     Parameters:
#     - input_json_path: Path to input JSON file with judge model scores.
#     - output_json_path: Path to output JSON file to save updated entities.
#     """
#     with open(input_json_path, "r", encoding="utf-8") as f:
#         entity_data_list = json.load(f)

#     clarity_scores = []
#     conciseness_scores = []
#     completeness_scores = []
#     for entity in entity_data_list:
#         # Collect scores by dimension
#         for key, value in entity['evaluations'].items():
#             clarity_scores.append(value['Clarity'])
#             conciseness_scores.append(value['Conciseness'])
#             completeness_scores.append(value['Completeness'])

#         # Compute mean and std
#         def calc_stats(scores):
#             return {
#                 "mean": float(np.mean(scores)) if scores else None,
#                 "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None
#             }

#         entity["stats"] = {
#             "Clarity": calc_stats(clarity_scores),
#             "Conciseness": calc_stats(conciseness_scores),
#             "Completeness": calc_stats(completeness_scores)
#         }


#     # Save updated JSON
#     with open(output_json_path, "w", encoding="utf-8") as f:
#         json.dump(entity_data_list, f, indent=2, ensure_ascii=False)

#     print(f"✅ Updated entities saved to: {output_json_path}")

# def update_mean_std_scores_in_json(input_json_path: str, output_json_path: str):
#     """
#     For each entity, compute mean and standard deviation across judge models
#     for Clarity, Conciseness, and Completeness. Save results in a 'stats' field.

#     Parameters:
#     - input_json_path: Path to input JSON file with judge model scores.
#     - output_json_path: Path to output JSON file to save updated entities.
#     """
#     with open(input_json_path, "r", encoding="utf-8") as f:
#         entity_data_list = json.load(f)

#     def calc_stats(scores):
#         return {
#             "mean": float(np.mean(scores)) if scores else None,
#             "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None
#         }

#     for entity in entity_data_list:
#         clarity_scores = []
#         conciseness_scores = []
#         completeness_scores = []

#         # Collect scores from all judge models for this entity
#         for evaluation in entity.get("evaluations", {}).values():
#             clarity_scores.append(evaluation.get("Clarity"))
#             conciseness_scores.append(evaluation.get("Conciseness"))
#             completeness_scores.append(evaluation.get("Completeness"))

#         # Store per-entity stats
#         entity["stats"] = {
#             "Clarity": calc_stats(clarity_scores),
#             "Conciseness": calc_stats(conciseness_scores),
#             "Completeness": calc_stats(completeness_scores)
#         }

#     # Save updated JSON
#     with open(output_json_path, "w", encoding="utf-8") as f:
#         json.dump(entity_data_list, f, indent=2, ensure_ascii=False)

#     print(f"✅ Per-entity stats saved to: {output_json_path}")

def update_mean_std_scores_in_json(input_json_path: str, output_json_path: str):
    """
    For each entity, compute statistics across judge models for Clarity, Conciseness, 
    and Completeness. Includes:
    - mean, std, min, max of valid scores
    - count of None values
    - percentage of valid scores
    - total count of scores
    
    Parameters:
    - input_json_path: Path to input JSON file with judge model scores.
    - output_json_path: Path to output JSON file to save updated entities.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        entity_data_list = json.load(f)

    def calc_stats(scores: List[Optional[float]]) -> Dict[str, Any]:
        valid_scores = [s for s in scores if s is not None]
        valid_count = len(valid_scores)
        total_count = len(scores)
        
        return {
            "mean": float(np.mean(valid_scores)) if valid_count > 0 else None,
            "std": float(np.std(valid_scores, ddof=1)) if valid_count > 1 else None,
            "min": float(min(valid_scores)) if valid_count > 0 else None,
            "max": float(max(valid_scores)) if valid_count > 0 else None,
            "count_none": total_count - valid_count,
            "count_total": total_count,
            "valid_percentage": round((valid_count / total_count) * 100, 2) if total_count > 0 else 0.0,
        }

    for entity in entity_data_list:
        clarity_scores = []
        conciseness_scores = []
        completeness_scores = []

        # Collect scores from all judge models for this entity
        for evaluation in entity.get("evaluations", {}).values():
            clarity_scores.append(evaluation.get("Clarity"))
            conciseness_scores.append(evaluation.get("Conciseness"))
            completeness_scores.append(evaluation.get("Completeness"))

        # Store per-entity stats
        entity["stats"] = {
            "Clarity": calc_stats(clarity_scores),
            "Conciseness": calc_stats(conciseness_scores),
            "Completeness": calc_stats(completeness_scores)
        }

    # Save updated JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entity_data_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Per-entity stats saved to: {output_json_path}")

    return entity_data_list


# def plot_evaluation_stats(input_data):
#     # Plot mean and standard deviation of evaluation scores from
#     # JSON file.
#     # Parameters:
#     # - input_data: List of entities with updated
#     # I BEDE
#     # stats.

#     entity_data_list = input_data
#     clarity_means = []
#     clarity_stds = []
#     conciseness_means = []
#     conciseness_stds = []
#     completeness_means = []
#     completeness_stds = []
    
#     for entity in entity_data_list:
#         clarity_means.append(entity["stats"]["Clarity"]["mean"])
#         clarity_stds.append(entity["stats"]["Clarity"]["std"])
#         conciseness_means.append(entity["stats"]["Conciseness"]["mean"])
#         conciseness_stds.append(entity["stats"]["Conciseness"]["std"])
#         completeness_means.append(entity["stats"]["Completeness"]["mean"]) 
#         completeness_stds.append (entity["stats"]["Completeness"]["std"])
        
#     # VisuaLization
#     labels = ['Clarity', 'Conciseness', 'Completeness']
#     means = [np.mean(clarity_means), np.mean(conciseness_means), np.mean(completeness_means)]
#     stds = [np.mean(clarity_stds), np.mean(conciseness_stds), np.mean(completeness_stds)]
    
#     x = np.arange(len(labels))
#     width = 0.35
    
#     fig, ax = plt.subplots()
#     bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=['skyblue', "lightgreen", "salmon"])
    
#     # Add Labels and title
#     ax.set_ylabel('Average Scone')
#     ax.set_title('Evaluation Scores: Mean and Variability')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.set_ylim(0, 6) # Assuming scores are between @ and 5
    
#     # Add annotations for mean and std
#     for i, bar in enumerate(bars):
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'Mean: {round(yval, 2)}\nStd: {round(stds[i], 2)}',
#                 ha='center', va='bottom', fontsize=9)
    
#     # CLean design
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            
#     plt.show()

def plot_evaluation_stats(filtered_data, selected_entity_ids):
    """
    Plot evaluation scores (mean + std) for selected entity_ids.

    Parameters:
    - filtered_data: list of dicts (each containing 'entity_id' and 'stats')
    - selected_entity_ids: list of entity IDs to plot
    """
    
    for entity in filtered_data:
        entity_id = entity["entity_id"]
        if entity_id not in selected_entity_ids:
            continue
        
        stats = entity["stats"]
        labels = ['Clarity', 'Conciseness', 'Completeness']
        means = [
            stats['Clarity']['mean'],
            stats['Conciseness']['mean'],
            stats['Completeness']['mean']
        ]
        stds = [
            stats['Clarity']['std'],
            stats['Conciseness']['std'],
            stats['Completeness']['std']
        ]

        x = np.arange(len(labels))
        width = 0.4
        # colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # teal, orange, blue
        colors = ['skyblue', "lightgreen", "salmon"]
        
        fig, ax = plt.subplots()
        bars = ax.bar(x, means, width, yerr=stds, capsize=6, color=colors)

        # Add labels and title
        ax.set_ylabel('Average Score')
        ax.set_title(f'Evaluation Scores for Entity ID {entity_id} : Mean and Variability')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 6)

        # Add annotation text
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1,
                    f'Mean: {round(yval, 2)}\nStd: {round(stds[i], 2)}',
                    ha='center', va='bottom', fontsize=9)

        # Clean design
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


def plot_entity_quality_stats_business(entity_data_list, threshold=4.0, save_path=None):
    entity_ids = list(range(1, len(entity_data_list) + 1))
    clarity_means = [e["stats"]["Clarity"]["mean"] for e in entity_data_list]
    clarity_stds = [e["stats"]["Clarity"]["std"] for e in entity_data_list]
    
    conciseness_means = [e["stats"]["Conciseness"]["mean"] for e in entity_data_list]
    conciseness_stds = [e["stats"]["Conciseness"]["std"] for e in entity_data_list]

    completeness_means = [e["stats"]["Completeness"]["mean"] for e in entity_data_list]
    completeness_stds = [e["stats"]["Completeness"]["std"] for e in entity_data_list]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    x = np.arange(len(entity_ids))

    def plot_metric(ax, means, stds, color, title):
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.75)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1-5 scale)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        for i, mean_score in enumerate(means):
            if mean_score < threshold:
                bars[i].set_color('red')

    plot_metric(axes[0], clarity_means, clarity_stds, 'dodgerblue', "Clarity Scores (Mean ± Std Dev)")
    plot_metric(axes[1], conciseness_means, conciseness_stds, 'mediumseagreen', "Conciseness Scores (Mean ± Std Dev)")
    plot_metric(axes[2], completeness_means, completeness_stds, 'darkorange', "Completeness Scores (Mean ± Std Dev)")

    axes[2].set_xlabel("Entity Number", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(entity_ids, rotation=45, ha='right')

    fig.suptitle("Narrative Quality Evaluation Across Entities\n(Mean Scores with Standard Deviation Error Bars)", fontsize=16, fontweight='bold', y=0.95)

    # Legend for color meanings
    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='black', label='Clarity Score ≥ Threshold'),
        Patch(facecolor='mediumseagreen', edgecolor='black', label='Conciseness Score ≥ Threshold'),
        Patch(facecolor='darkorange', edgecolor='black', label='Completeness Score ≥ Threshold'),
        Patch(facecolor='red', edgecolor='black', label='Score < Threshold'),
        Patch(facecolor='none', edgecolor='red', linestyle='--', label='Threshold Line')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.show()



def get_entities_below_threshold(entity_data_list, threshold=4.0):
    below_threshold_entities = []

    for i, entity in enumerate(entity_data_list, start=1):
        clarity = entity["stats"]["Clarity"]["mean"]
        conciseness = entity["stats"]["Conciseness"]["mean"]
        completeness = entity["stats"]["Completeness"]["mean"]

        if (clarity is not None and clarity < threshold) or \
           (conciseness is not None and conciseness < threshold) or \
           (completeness is not None and completeness < threshold):
            below_threshold_entities.append(i)  # or entity ID if you have one

    return below_threshold_entities


def plot_entity_quality_stats_business_curve(entity_data_list, threshold=4.0, save_path=None):
    # Prepare data
    entity_ids = [e['entity_id'] for e in entity_data_list]  # Use actual entity IDs
    metrics = {
        'Clarity': {'color': 'dodgerblue', 'title': "Clarity Scores"},
        'Conciseness': {'color': 'mediumseagreen', 'title': "Conciseness Scores"}, 
        'Completeness': {'color': 'darkorange', 'title': "Completeness Scores"}
    }
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Generate smooth x-values for curves
    x = np.arange(len(entity_ids))
    x_smooth = np.linspace(x.min(), x.max(), 300)  # 300 points for smooth curve
    
    def plot_metric(ax, metric, means, stds):
        # Create smooth curves using spline interpolation
        spl = make_interp_spline(x, means, k=3)  # Cubic spline
        means_smooth = spl(x_smooth)
        
        spl_upper = make_interp_spline(x, np.array(means)+np.array(stds), k=3)
        upper_smooth = spl_upper(x_smooth)
        
        spl_lower = make_interp_spline(x, np.array(means)-np.array(stds), k=3)
        lower_smooth = spl_lower(x_smooth)
        
        # Plot smooth curve and confidence band
        ax.plot(x_smooth, means_smooth, 
                color=metric['color'], 
                linewidth=2.5,
                label='Mean Score')
        
        ax.fill_between(x_smooth, lower_smooth, upper_smooth,
                       color=metric['color'], alpha=0.2,
                       label='±1 Std Dev')
        
        # Add original data points
        ax.scatter(x, means, 
                  color=metric['color'],
                  s=80, zorder=3,
                  edgecolor='white', linewidth=1)
        
        # Threshold line and styling
        ax.axhline(threshold, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold})')
        
        # Highlight points below threshold
        below_threshold = np.array(means) < threshold
        ax.scatter(x[below_threshold], np.array(means)[below_threshold],
                 color='red', s=100, zorder=4,
                 edgecolor='black', linewidth=1)
        
        # Customize axes
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1-5)", fontsize=11)
        ax.set_title(metric['title'], fontsize=12, pad=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    # Plot each metric
    for ax, (metric_name, metric_style) in zip(axes, metrics.items()):
        means = [e["stats"][metric_name]["mean"] for e in entity_data_list]
        stds = [e["stats"][metric_name]["std"] for e in entity_data_list]
        plot_metric(ax, metric_style, means, stds)
    
    # X-axis formatting with slight tilt (30 degrees)
    axes[-1].set_xticks(np.arange(len(entity_ids)))
    axes[-1].set_xticklabels(entity_ids, rotation=30, ha='right', fontsize=10)  # 30 degree tilt
    axes[-1].set_xlabel("Entity ID", fontsize=11)
    
    # Add slight padding below x-axis labels
    plt.subplots_adjust(bottom=0.12)
    
    # Main title
    fig.suptitle("Narrative Quality Evaluation Trends\n(Smooth Curve with Standard Deviation Band)", 
                fontsize=14, fontweight='bold', y=0.97)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def plot_model_metric_trends(model_data_dict, save_prefix="model_metrics"):
    """
    Plots metric trends for multiple generation models across three quality dimensions.
    
    Args:
        model_data_dict: Dictionary containing model data (format shown in sample)
        save_prefix: Prefix for saving figures (optional)
    """
    # Prepare metrics and colors
    metrics = ['Clarity', 'Conciseness', 'Completeness']
    colors = plt.cm.tab10.colors  # Distinct colors for models
    
    # Create one figure per metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Plot each model's trend
        for model_idx, (model_name, entity_data) in enumerate(model_data_dict.items()):
            # Extract data for this model
            entity_ids = [e['entity_id'] for e in entity_data]
            means = [e['stats'][metric]['mean'] for e in entity_data]
            stds = [e['stats'][metric]['std'] for e in entity_data]
            
            # Create smooth curve
            x = np.arange(len(entity_ids))
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, means, k=3)
            means_smooth = spl(x_smooth)
            
            # Plot main line and confidence band
            plt.plot(x_smooth, means_smooth, 
                    color=colors[model_idx], 
                    linewidth=2.5,
                    label=model_name)
            
            plt.fill_between(x_smooth, 
                            means_smooth - np.mean(stds), 
                            means_smooth + np.mean(stds),
                            color=colors[model_idx], 
                            alpha=0.15)
            
            # Add actual data points
            plt.scatter(x, means, 
                       color=colors[model_idx],
                       s=80, zorder=3,
                       edgecolor='white', linewidth=1)
        
        # Customize plot
        plt.title(f'{metric} Scores Across Entities by Generation Model', fontsize=14)
        plt.xlabel('Entity ID', fontsize=12)
        plt.ylabel('Score (1-5 scale)', fontsize=12)
        plt.xticks(np.arange(len(entity_ids)), labels=entity_ids, rotation=45, ha='right')
        plt.ylim(0, 5.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Adjust layout and save if needed
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_{metric.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()


# def plot_model_metric_trends_combined(model_data_dict, save_path="plots/model_metrics_combined.png"):
#     """
#     Plots metric trends for multiple generation models in a single figure with subplots.

#     Args:
#         model_data_dict (dict): Dictionary of model names to entity data.
#         save_path (str): Full file path to save the figure.
#     """
#     metrics = ['Clarity', 'Conciseness', 'Completeness']
#     colors = plt.cm.tab10.colors  # up to 10 distinct model lines

#     # Prepare figure and axes
#     fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
#     for ax, metric in zip(axes, metrics):
#         for model_idx, (model_name, entity_data) in enumerate(model_data_dict.items()):
#             entity_ids = [e['entity_id'] for e in entity_data]
#             means = [e['stats'][metric]['mean'] for e in entity_data]
#             stds = [e['stats'][metric]['std'] for e in entity_data]

#             x = np.arange(len(entity_ids))
#             if len(x) >= 4:  # Spline needs at least k+1 points
#                 x_smooth = np.linspace(x.min(), x.max(), 300)
#                 spl = make_interp_spline(x, means, k=3)
#                 means_smooth = spl(x_smooth)
#             else:
#                 x_smooth = x
#                 means_smooth = means

#             ax.plot(x_smooth, means_smooth, 
#                     color=colors[model_idx], 
#                     linewidth=2.5,
#                     label=model_name)
            
#             ax.fill_between(x_smooth,
#                             np.array(means_smooth) - np.mean(stds),
#                             np.array(means_smooth) + np.mean(stds),
#                             color=colors[model_idx],
#                             alpha=0.15)
            
#             ax.scatter(x, means,
#                        color=colors[model_idx],
#                        s=80, zorder=3,
#                        edgecolor='white', linewidth=1)

#         ax.set_title(f'{metric}', fontsize=14)
#         ax.set_xlabel('Entity Index', fontsize=12)
#         ax.set_ylim(0, 5.5)
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(axis='x', rotation=45)

#     axes[0].set_ylabel('Score (1–5 scale)', fontsize=12)
#     axes[-1].legend(loc='upper right', fontsize=10)

#     fig.suptitle('Model Metric Trends by Generation Model', fontsize=16)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)

#     print(f"Combined figure saved to: {save_path}")

def plot_model_metric_trends_combined_vertical(model_data_dict, save_path="plots/model_metrics_vertical.png"):
    """
    Plots metric trends for multiple generation models in a vertically stacked figure.
    
    Args:
        model_data_dict (dict): Mapping model names to list of entity dicts with evaluation stats.
        save_path (str): File path where the figure will be saved.
    """
    metrics = ['Clarity', 'Conciseness', 'Completeness']
    colors = plt.cm.tab10.colors  # distinct colors per model

    # Set up 3-row subplot (vertical stacking)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 16), sharex=True)

    for ax, metric in zip(axes, metrics):
        for model_idx, (model_name, entity_data) in enumerate(model_data_dict.items()):
            entity_ids = [e['entity_id'] for e in entity_data]
            means = [e['stats'][metric]['mean'] for e in entity_data]
            stds = [e['stats'][metric]['std'] for e in entity_data]

            x = np.arange(len(entity_ids))
            if len(x) >= 4:
                x_smooth = np.linspace(x.min(), x.max(), 300)
                spl = make_interp_spline(x, means, k=3)
                means_smooth = spl(x_smooth)
            else:
                x_smooth = x
                means_smooth = means

            ax.plot(x_smooth, means_smooth,
                    color=colors[model_idx],
                    linewidth=2.5,
                    label=model_name)

            ax.fill_between(x_smooth,
                            np.array(means_smooth) - np.mean(stds),
                            np.array(means_smooth) + np.mean(stds),
                            color=colors[model_idx],
                            alpha=0.15)

            ax.scatter(x, means,
                       color=colors[model_idx],
                       s=80, zorder=3,
                       edgecolor='white', linewidth=1)

        ax.set_title(f'{metric} Score', fontsize=14)
        ax.set_ylabel('Score (1–5 scale)', fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Entity Index', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    fig.suptitle('Model Metric Trends by Generation Model', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    # Show plot
    plt.show()


# def print_business_friendly_summary(entity):
#     from tabulate import tabulate

#     entity_id = entity['entity_id']
#     risk_score = entity['risk_score']
#     prompt = entity.get('prompt', '[No prompt]')
#     llm_explanation = entity.get('llm_explanation', '[No explanation]')
#     stats = entity.get('stats', {})
#     features = entity['features']

#     # Extract top 5 features by contribution
#     feature_rows = sorted([
#         (
#             fname,
#             fdata.get('feature_value'),
#             fdata.get('contribution_pct', 0)
#         )
#         for fname, fdata in features.items()
#     ], key=lambda x: -x[2])[:5]

#     # Format feature table
#     feature_table = tabulate(
#         [(fname, fvalue, f"{contrib:.1f}%") for fname, fvalue, contrib in feature_rows],
#         headers=["Feature", "Value", "Contribution %"],
#         tablefmt="grid"
#     )

#     # Format stats table
#     stats_table = tabulate(
#         [
#             (metric,
#              f"{metric_stats['mean']:.1f}",
#              f"{metric_stats['std']:.2f}",
#              f"{metric_stats['min']:.1f}–{metric_stats['max']:.1f}")
#             for metric, metric_stats in stats.items()
#         ],
#         headers=["Metric", "Mean", "Std Dev", "Range"],
#         tablefmt="grid"
#     )

#     # Print the full report
#     print(f"\n{'='*80}")
#     print(f"🆔 Entity ID: {entity_id} | 💡 Risk Score: {risk_score:.0%}")
#     print(f"{'='*80}")

#     print("\n📌 Top Contributing Features:")
#     print(feature_table)

#     print("\n🤖 Prompt Given to LLM:")
#     print(prompt)

#     print("\n📝 LLM Explanation:")
#     print(llm_explanation)

#     print("\n📊 Evaluation Summary:")
#     print(stats_table)

#     # Optional: Flag discrepancy between risk score and explanation (if numeric in text)
#     import re
#     match = re.search(r'(\d{2,3})\%', llm_explanation)
#     if match:
#         explained_score = int(match.group(1))
#         actual_score = int(risk_score * 100)
#         if abs(explained_score - actual_score) > 20:
#             print(f"\n⚠️ Warning: LLM explanation mentions a risk score of {explained_score}%, "
#                   f"but actual score is {actual_score}%. This may indicate a hallucination.")

def print_business_friendly_summary(entity):
    from tabulate import tabulate
    import re

    entity_id = entity['entity_id']
    risk_score = entity['risk_score']
    prompt = entity.get('prompt', '[No prompt]')
    llm_explanation = entity.get('llm_explanation', '[No explanation]')
    stats = entity.get('stats', {})
    features = entity['features']

    # Extract top 5 features by contribution
    feature_rows = sorted([
        (fname, fdata.get('contribution_pct', 0))
        for fname, fdata in features.items()
    ], key=lambda x: -x[1])[:5]

    # Format feature table without value
    feature_table = tabulate(
        [(fname, f"{contrib:.1f}%") for fname, contrib in feature_rows],
        headers=["Feature", "Contribution %"],
        tablefmt="grid"
    )

    # Format stats table
    stats_table = tabulate(
        [
            (metric,
             f"{metric_stats['mean']:.1f}",
             f"{metric_stats['std']:.2f}",
             f"{metric_stats['min']:.1f}–{metric_stats['max']:.1f}")
            for metric, metric_stats in stats.items()
        ],
        headers=["Metric", "Mean", "Std Dev", "Range"],
        tablefmt="grid"
    )

    # Print the full report
    print(f"\n{'='*80}")
    # print(f"🆔 Entity ID: {entity_id} | 💡 Risk Score: {risk_score:.0%}")
    print(f" Entity ID: {entity_id} |  Risk Score: {risk_score:.0%}")
    print(f"{'='*80}")

    # print("\n📌 Top Contributing Features:")
    print("\n Top Contributing Features:")
    print(feature_table)

    # print("\n🤖 Prompt Given to LLM:")
    print("\n Prompt Given to LLM:")
    print(prompt)

    # print("\n📝 LLM Explanation:")
    print("\n LLM Explanation:")
    print(llm_explanation)

    # print("\n📊 Evaluation Summary:")
    print("\n Evaluation Summary:")
    print(stats_table)

    # Optional: Flag discrepancy between risk score and explanation (if numeric in text)
    match = re.search(r'(\d{2,3})\%', llm_explanation)
    if match:
        explained_score = int(match.group(1))
        actual_score = int(risk_score * 100)
        if abs(explained_score - actual_score) > 20:
            print(f"\n⚠️ Warning: LLM explanation mentions a risk score of {explained_score}%, "
                  f"but actual score is {actual_score}%. This may indicate a hallucination.")







def generate_controlled_narratives(
    input_json_path: str,
    output_json_path: str,
    feature_library_df: pd.DataFrame,
    azure_client: AzureClient,
    ccc_levels: dict = {"clarity": 3, "conciseness": 3, "completeness": 3},
    top_n: int = 10,
    log_every_n: int = 5
):
    """
    Generates LLM narratives with controlled Clarity/Conciseness/Completeness levels.

    Adds prompt and llm_explanation fields to each entity.

    Parameters:
    - input_json_path: JSON input file with entity features
    - output_json_path: JSON file to write enriched entities
    - feature_library_df: DataFrame with feature_name and description
    - azure_client: AzureClient instance to call LLM
    - ccc_levels: Dict with keys clarity, conciseness, completeness (values 1–5)
    - top_n: Number of top features to include
    - log_every_n: Frequency for logging
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    for i, entity in enumerate(entities):
        prompt = build_controlled_prompt_from_entity_row(
            entity_data=entity,
            feature_library_df=feature_library_df,
            ccc_levels=ccc_levels,
            top_n=top_n
        )
        explanation = azure_client.get_response(prompt)

        entity["prompt"] = prompt
        entity["llm_explanation"] = explanation

        if (i + 1) % log_every_n == 0 or i == len(entities) - 1:
            print(f"📝 Generated explanation for {i + 1}/{len(entities)} entities")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Narratives saved to {output_json_path}")


def evaluate_controlled_narratives(
    input_json_path: str,
    output_json_path: str,
    azure_client: AzureClient,
    judge_models: dict = JUDGE_MODELS,
    log_every_n: int = 5
):
    """
    Evaluates LLM-generated narratives for CCC quality using judge models.

    Parameters:
    - input_json_path: JSON file with 'prompt' and 'llm_explanation'
    - output_json_path: Path to write evaluations
    - azure_client: AzureClient instance to call judge LLMs
    - judge_models: Dict of judge model name -> deployment name
    - log_every_n: Logging frequency
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    for i, entity in enumerate(entities):
        prompt = entity.get("prompt")
        explanation = entity.get("llm_explanation")

        if not prompt or not explanation:
            print(f"⚠️ Missing prompt or explanation for entity {i + 1}. Skipping.")
            continue

        entity["evaluations"] = {}

        for judge_name, deployment_name in judge_models.items():
            eval_prompt = build_evaluation_prompt(prompt, explanation)

            start_time = datetime.now()
            eval_text = azure_client.get_response(eval_prompt, deployment_name=deployment_name)
            end_time = datetime.now()

            clarity, conciseness, completeness = extract_scores(eval_text)
            eval_duration = (end_time - start_time).total_seconds()

            entity["evaluations"][judge_name] = {
                "Clarity": clarity,
                "Conciseness": conciseness,
                "Completeness": completeness,
                "Summary": eval_text.strip(),
                "EvalTime": eval_duration
            }

        if (i + 1) % log_every_n == 0 or i == len(entities) - 1:
            print(f"📊 Evaluated {i + 1}/{len(entities)} entities")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluations saved to {output_json_path}")



def generate_and_evaluate_controlled_narratives(
    input_json_path: str,
    output_json_path: str,
    feature_library_df: pd.DataFrame,
    azure_client: AzureClient,
    judge_models: dict = JUDGE_MODELS,
    control_sets: list = None,
    top_n: int = 10,
    log_every_n: int = 5
):
    """
    Orchestrates full workflow: controlled prompt → controlled narrative → judge model evaluation.

    Parameters:
    - input_json_path: JSON input file with 'prompt' and 'llm_explanation'
    - output_json_path: Output path to save results
    - feature_library_df: DataFrame containing feature descriptions
    - azure_client: LLM caller
    - judge_models: Dict of judge model name -> deployment name
    - control_sets: List of CCC dicts, e.g. [{'clarity': 5, 'conciseness': 4, 'completeness': 5}, ...]
    - top_n: Number of features to include in the prompt
    - log_every_n: Frequency of logging
    """
    if control_sets is None:
        control_sets = [
            {"clarity": 3, "conciseness": 3, "completeness": 3},
            {"clarity": 5, "conciseness": 4, "completeness": 5}
        ]

    with open(input_json_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    output_entities = []

    for idx, entity in enumerate(entities):
        base_prompt = entity.get("prompt")
        base_explanation = entity.get("llm_explanation")
        entity_id = entity.get("entity_id")

        if not base_prompt or not base_explanation or not entity_id:
            print(f"⚠️ Skipping entity {idx+1}: missing prompt/explanation/entity_id")
            continue

        entity_output = {
            "entity_id": entity_id,
            "prompt": base_prompt,
            "llm_explanation": base_explanation,
            "controlled_variants": []
        }

        for control in control_sets:
            controlled_prompt = build_controlled_prompt_from_entity_row(
                entity_data=entity,
                feature_library_df=feature_library_df,
                ccc_levels=control,
                top_n=top_n
            )
            controlled_narrative = azure_client.get_response(controlled_prompt)

            variant = {
                "ccc_values": control,
                "controlled_prompt": controlled_prompt,
                "controlled_narrative": controlled_narrative,
                "evaluations": {}
            }

            for judge_name, deployment_name in judge_models.items():
                eval_prompt = build_evaluation_prompt(controlled_prompt, controlled_narrative)

                start_time = datetime.now()
                eval_text = azure_client.get_response(eval_prompt, deployment_name=deployment_name)
                end_time = datetime.now()

                clarity, conciseness, completeness = extract_scores(eval_text)
                eval_duration = (end_time - start_time).total_seconds()

                variant["evaluations"][judge_name] = {
                    "Clarity": clarity,
                    "Conciseness": conciseness,
                    "Completeness": completeness,
                    "Summary": eval_text.strip(),
                    "EvalTime": eval_duration
                }

            entity_output["controlled_variants"].append(variant)

        output_entities.append(entity_output)

        if (idx + 1) % log_every_n == 0 or idx == len(entities) - 1:
            print(f"✅ Processed {idx + 1}/{len(entities)} entities")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_entities, f, indent=2, ensure_ascii=False)

    print(f"🎯 All controlled prompts, narratives, and evaluations saved to {output_json_path}")


def generate_plain_explanations(
    input_excel_path: str,
    output_excel_path: str,
    azure_client,
    model_name: str = "your-deployment-name"
):
    # Load Excel
    df = pd.read_excel(input_excel_path)

    # Add new column for LLM explanations
    explanations = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        feature = row["feature"]
        meaning = row["meaning"]

        prompt = (
            f"You are a data analyst explaining a feature in simple terms.\n"
            f"Feature: {feature}\n"
            f"Meaning: {meaning}\n"
            f"Write a plain-language explanation between 30 and 50 words so a business audience can understand it clearly:"
        )

        response = azure_client.get_response(prompt, model_name=model_name)
        explanations.append(response.strip())

    df["llm_explanation"] = explanations

    # Save to Excel
    df.to_excel(output_excel_path, index=False)
    print(f"✅ Explanations saved to: {output_excel_path}")


def show_dict_structure(d, indent=0):
    for key, value in d.items():
        print("  " * indent + f"├── {key} ({type(value).__name__})")
        if isinstance(value, dict):
            show_dict_structure(value, indent + 1)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            print("  " * (indent + 1) + f"└── [nested dicts]")
            show_dict_structure(value[0], indent + 2)


def display_dict_structure(d, indent=4, level=0):
    indent_str = " " * indent * level
    next_level = level + 1
    print(indent_str + "{")
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{indent_str}    \"{key}\": {{")
            display_dict_structure(value, indent, next_level)
            print(indent_str + "    },")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{indent_str}    \"{key}\": [")
            print(f"{indent_str}        {{")
            display_dict_structure(value[0], indent, next_level + 1)
            print(f"{indent_str}        }},")
            print(f"{indent_str}    ],")
        else:
            print(f"{indent_str}    \"{key}\": {repr(value)},")
    print(indent_str + "}")

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
        input_json_path="entities_with_llm_explanations.json",
        output_json_path="entities_enriched_evaluated.json",
        feature_library_df=feature_library_df,
        azure_client=azure_client,
        judge_models=judge_models,
        top_n=5,
        log_every_n=5  # print progress every 3 entities
    )
    
    update_mean_std_scores_in_json(
        input_json_path="entities_enriched_evaluated.json",
        output_json_path="entities_with_stats.json"
    )

    plot_entity_quality_stats_business(output, threshold=4.0, save_path="../data/output/evaluation_scores.png")

    # Example usage:
    below_thresh_ids = get_entities_below_threshold(output, threshold=4.0)
    print("Entities below threshold:", below_thresh_ids)

    generate_and_evaluate_controlled_narratives(
        input_json_path="input.json",
        output_json_path="controlled_output.json",
        feature_library_df=feature_library_df,
        azure_client=azure_client,
        control_sets=[
            {"clarity": 1, "conciseness": 1, "completeness": 1},
            {"clarity": 3, "conciseness": 3, "completeness": 3}
        ],
        top_n=10
    )
