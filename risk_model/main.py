from config import *
from data.data_loader import generate_synthetic_data
from data.feature_library import load_feature_descriptions
from model.trainer import train_xgboost
from model.explainer import compute_shap_values
from narrative.prompt_generator import build_prompt
from narrative.azure_openai_client import get_azure_openai_client, generate_narrative
from tabulate import tabulate
import pandas as pd

def main():
    # Step 1: Load data
    X, y = generate_synthetic_data()
    
    # Step 2: Train model
    model, X_test = train_xgboost(X, y)
    
    # Step 3: Explain one example
    entity_index = 0
    entity_features = X_test.iloc[[entity_index]]
    risk_score = model.predict(entity_features)[0]
    
    # Step 4: SHAP values
    top_features = compute_shap_values(model, X, entity_features, TOP_N_FEATURES)
    
    # Step 5: Feature descriptions
    feature_descriptions = load_feature_descriptions(feature_names=X.columns)
    
    # Step 6: Build prompt
    prompt = build_prompt(risk_score, top_features, feature_descriptions)
    print("\n=== PROMPT TO LLM ===\n")
    print(prompt)
    
    # Step 7: Azure OpenAI call
    client = get_azure_openai_client(AZURE_OPENAI_ENDPOINT)
    narrative = generate_narrative(client, DEPLOYMENT_NAME, prompt)
    
    # Step 8: Output summary
    summary_table = pd.DataFrame({
        "Entity Index": [entity_index],
        "Predicted Risk Score (%)": [round(risk_score * 100, 2)],
        "Top Features": [", ".join(top_features['Feature'])],
        "Narrative": [narrative],
        "Evaluation Score": ["N/A (manual review recommended)"]
    })
    
    print("\n=== SUMMARY TABLE ===\n")
    print(tabulate(summary_table, headers="keys", tablefmt="fancy_grid", showindex=False))

if __name__ == "__main__":
    main()
