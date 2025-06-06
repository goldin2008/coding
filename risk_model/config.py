import os

# ===============================
# Azure OpenAI Configuration
# ===============================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "your-gpt-deployment")

# ===============================
# Data Generation Parameters
# ===============================
AML_DATA_PATH = "data/aml_data.csv"
NUM_SAMPLES = 1000
FRAUD_RATIO = 0.3  # Percentage of fraud/risk cases in generated data

# ===============================
# Feature Library
# ===============================
FEATURE_LIBRARY_CSV_PATH = "data/feature_library.csv"

# ===============================
# Dummy Dataset
# ===============================
DUMMY_DATA_CSV = "data/dummy_aml_data.csv"

# ===============================
# XGBoost Model Parameters
# ===============================
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42
}

OUTPUT_MODEL_PATH = "models/xgb_model.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ===============================
# SHAP Parameters
# ===============================
TOP_N_FEATURES = 10  # Number of top features to include in explanation

# ===============================
# Prompt Generator Parameters
# ===============================
RISK_SCORE_DECIMALS = 2  # Number of decimal points for risk score
CONTRIBUTION_PERCENT_DECIMALS = 1  # Number of decimal points for contribution percentages

# ===============================
# Model Training/Testing
# ===============================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===============================
# Notebook/Module Control Flags
# ===============================
PLOT_SHAP_SUMMARY = True  # Whether to plot SHAP summary plot

# ===============================
# Miscellaneous
# ===============================
ENTITY_INDEX_FOR_EXPLANATION = 0  # Index of the entity to explain in the test set
