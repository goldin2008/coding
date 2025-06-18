import os
from dotenv import load_dotenv

# Load environment variables (if .env file is used)
load_dotenv()


# ===============================
# Azure OpenAI Configuration
# ===============================
AZURE_OPENAI = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key"),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "your-gpt-deployment"),
}

AZURE_CONFIG = {

}

# ===============================
# OpenAI LLM Configuration
# ===============================
LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model_name": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

PROMPT_CONFIG = {
    "content": "You are a helpful assistant. Please provide concise explanations."
}

# ===============================
# Data Generation & Dataset Paths
# ===============================
DATA_PATHS = {
    "aml_data": "data/aml_data.csv",
    "feature_library_csv": "data/feature_library.csv",
    "dummy_data_csv": "data/dummy_aml_data.csv",
}

DATA_GENERATION = {
    "num_samples": 1000,
    "fraud_ratio": 0.3, # Percentage of fraud/risk cases in generated data
}

# ===============================
# Model Training and Evaluation
# ===============================
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": TRAINING_CONFIG["random_state"],
}

OUTPUT_MODEL_PATH = "models/xgb_model.json"

# ===============================
# SHAP Configuration
# ===============================
SHAP_CONFIG = {
    "top_n_features": 10,
    "plot_summary": True,
}

# Clustering configuration
CLUSTER_CONFIG = {
    "n_clusters": 3,
    "random_state": 42
}

# ===============================
# Prompt Generator Parameters
# ===============================
PROMPT_PARAMS = {
    "risk_score_decimals": 2, # Number of decimal points for risk score
    "contribution_percent_decimals": 1, # Number of decimal points for contribution percentages
}

# ===============================
# Notebook/Module Control Flags
# ===============================
PLOT_SHAP_SUMMARY = True  # Whether to plot SHAP summary plot

# ===============================
# Miscellaneous
# ===============================
ENTITY_INDEX_FOR_EXPLANATION = 0 # Index of the entity to explain in the test set