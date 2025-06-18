risk_model/
│
├── __init__.py
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing utilities
│   ├── feature_library.py      # Feature library and descriptions
│
├── model/
│   ├── __init__.py
│   ├── trainer.py              # XGBoost training pipeline
│   ├── shap_utils.py           # SHAP calculation helpers
│   ├── shap_analysis.py        # SHAP analysis and visualization
│
├── narrative/
│   ├── __init__.py
│   ├── prompt_generator.py     # Generates prompts for LLM explanations
│   ├── azure_openai_client.py  # Azure OpenAI API client
│
├── main.py                     # Entry point for training and explanation
│
├── config.py                   # Centralized configuration (paths, hyperparameters)
│
└── requirements.txt            # Python dependencies
