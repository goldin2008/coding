risk_explainer/
├── config/                        # All configuration constants
│   ├── __init__.py
│   └── config.py

├── clients/                       # LLM and judge clients
│   ├── __init__.py
│   └── azure_openai_client.py           # AzureClient and MockAzureClient

├── prompts/                       # Prompt builders
│   ├── __init__.py
│   └── prompt_generator.py        # Functions to build standard & controlled prompts

├── model/                         # Unit tests for each module
│   ├── __init__.py
│   ├── shap_analysis.py
│   ├── shap_utils.py
│   └── trainer.py

├── evaluation/                    # Judge model evaluation logic
│   ├── __init__.py
│   └── judge.py                  # extract_scores, evaluation prompt builder, etc.

├── workflows/                     # Main logic for generating and evaluating narratives
│   ├── __init__.py
│   └── generate_data.py         # load, build, evaluate, and save enriched JSONs

├── notbooks/                         # Jupyter notebooks for EDA, dev, demo
│   ├── 01_feature_analysis.ipynb       # Exploratory Data Analysis (EDA) of input features
│   ├── 02_llm_narrative_testing.ipynb  # Interactive narrative generation and tweaking
│   ├── 03_judge_eval_debug.ipynb       # Test judge evaluation on sample prompts
│   ├── 04_end_to_end_pipeline.ipynb    # Run full workflow interactively
│   └── 05_run.ipynb    # Run full workflow interactively

├── data/                          # Input/output files (not tracked in version control)
│   ├── input/
│   └── output/

├── tests/                         # Unit tests for each module
│   ├── __init__.py
│   ├── test_azure_client.py
│   ├── test_builders.py
│   ├── test_judge.py
│   └── test_generate_data.py

├── scripts/                       # Optional CLI or orchestration scripts
│   └── main.py                   # Entrypoint: load config, run full workflow

├── .env                           # Used with dotenv to manage secrets
├── .gitignore                    # Ignore sensitive/log/compiled/output files
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview, how to run, etc.
└── setup.py                      # Optional: make it pip-installable
