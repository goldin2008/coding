import os

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "your-gpt-deployment")
TOP_N_FEATURES = 5
