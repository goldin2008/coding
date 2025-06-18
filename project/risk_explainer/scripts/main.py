#!/usr/bin/env python3
"""
main.py - Entry point for the risk explanation narrative generation and evaluation pipeline.
"""

import json
import shap
from pathlib import Path

import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import MOCK_JUDGE_MODELS
from clients.azure_openai_client import get_azure_client
from evaluation import judge
import workflows.generate_data as gd
import prompts.prompt_generator as pg


def generate_model_results(model_name, feature_lib, azure_client, top_n):
    # Generate LLM explanations
    print("\n=== Generating LLM Explanations ===")
    updated_entities = gd.enrich_entities_with_llm_explanations(
        input_json_path="data/output/entities_with_prompts.json",
        output_json_path=f"data/output/entities_with_llm_explanations_{model_name}.json",
        feature_library_df=feature_lib,
        azure_client=azure_client,
        top_n=top_n,
        log_every_n=25
    )
    
    # Evaluate explanations with judge models
    print("\n=== Evaluating Explanations ===")
    enriched_entities = gd.enrich_and_evaluate_entities(
        input_json_path=f"data/output/entities_with_llm_explanations_{model_name}.json",
        output_json_path=f"data/output/entities_enriched_evaluated_{model_name}.json",
        feature_library_df=feature_lib,
        azure_client=azure_client,
        judge_models=judge.MOCK_JUDGE_MODELS,
        top_n=top_n,
        log_every_n=25
    )
    
    # Compute statistics
    print("\n=== Calculating Evaluation Statistics ===")
    entity_data_list = gd.update_mean_std_scores_in_json(
        input_json_path=f"data/output/entities_enriched_evaluated_{model_name}.json",
        output_json_path=f"data/output/entities_with_stats_{model_name}.json"
    )

    return entity_data_list

    
def main():
    # Configuration
    use_mock = True  # Set to False to use real Azure OpenAI
    top_n = 5
    threshold = 4.0  # Score threshold for identifying low-quality explanations
    
    # Initialize Azure client (mock or real)
    azure_client = get_azure_client(use_mock=use_mock)
    
    # Create feature library
    print("\n=== Creating AML Feature Library ===")
    feature_lib = gd.create_realistic_aml_feature_library()
    
    # Generate structured dataset with SHAP values
    print("\n=== Generating Structured Dataset ===")
    input_data_path = "data/input/structured_aml_dataset.json"
    Path("data/input").mkdir(parents=True, exist_ok=True)
    df_structured = gd.generate_structured_shap_dataset(
        feature_lib, 
        n_entities=100, 
        output_path=input_data_path
    )
    
    # Example SHAP visualization for entity 1
    print("\n=== Generating SHAP Explanation for Entity 1 ===")
    explanation = gd.load_shap_explanation(input_data_path, entity_id=1)
    # shap.plots.bar(explanation)
    
    # Generate prompt for first entity as example
    print("\n=== Sample Prompt Generation ===")
    with open(input_data_path, "r") as f:
        entity_data = json.loads(f.readline())  # Get first entity
    
    prompt = pg.build_feature_contribution_prompt_from_structured_json(
        entity_data, 
        feature_lib, 
        top_n=top_n
    )
    print("\nSample prompt for first entity:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # Add prompts to all entities
    print("\n=== Adding Prompts to All Entities ===")
    with open(input_data_path, "r") as f:
        entity_data_list = [json.loads(line) for line in f]
    
    df_with_prompts = gd.add_prompt_to_entity_json(
        entity_data_list, 
        feature_lib, 
        top_n=top_n
    )
    
    # Save entities with prompts
    output_data_path = "data/output/entities_with_prompts.json"
    Path("data/output").mkdir(parents=True, exist_ok=True)
    gd.save_entities_to_json(df_with_prompts, output_data_path)

    final = dict()

    models = ["model_1", "model_2", "model_3", "model_4", "model_5"]
    for model_name in models:
        entity_data_list = generate_model_results(model_name, feature_lib, azure_client, top_n)
        final[model_name] = entity_data_list

    output_data_path_final = "data/output/final.json"
    gd.save_entities_to_json(final, output_data_path_final)
    
    
    # # Generate LLM explanations
    # print("\n=== Generating LLM Explanations ===")
    # updated_entities = gd.enrich_entities_with_llm_explanations(
    #     input_json_path="data/output/entities_with_prompts.json",
    #     output_json_path="data/output/entities_with_llm_explanations.json",
    #     feature_library_df=feature_lib,
    #     azure_client=azure_client,
    #     top_n=top_n,
    #     log_every_n=25
    # )
    
    # # Evaluate explanations with judge models
    # print("\n=== Evaluating Explanations ===")
    # enriched_entities = gd.enrich_and_evaluate_entities(
    #     input_json_path="data/output/entities_with_llm_explanations.json",
    #     output_json_path="data/output/entities_enriched_evaluated.json",
    #     feature_library_df=feature_lib,
    #     azure_client=azure_client,
    #     judge_models=judge.MOCK_JUDGE_MODELS,
    #     top_n=top_n,
    #     log_every_n=25
    # )
    
    # # Compute statistics
    # print("\n=== Calculating Evaluation Statistics ===")
    # entity_data_list = gd.update_mean_std_scores_in_json(
    #     input_json_path="data/output/entities_enriched_evaluated.json",
    #     output_json_path="data/output/entities_with_stats.json"
    # )



    
    # # Load final output for visualization
    # with open("data/output/entities_with_stats.json", "r") as f:
    #     output = json.load(f)
    
    # # Identify and visualize low-quality explanations
    # print("\n=== Identifying Low-Quality Explanations ===")
    # below_thresh_ids = gd.get_entities_below_threshold(output, threshold=threshold)
    # print(f"Entities below threshold ({threshold}): {below_thresh_ids}")
    
    # if below_thresh_ids:
    #     print("\n=== Visualizing Low-Quality Explanations ===")
    #     filtered_data = [entity for entity in output if entity['entity_id'] in below_thresh_ids]
    #     gd.plot_entity_quality_stats_business(
    #         filtered_data, 
    #         threshold=threshold, 
    #         save_path="data/output/evaluation_scores.png"
    #     )
    # else:
    #     print("No entities below threshold found.")
        
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()