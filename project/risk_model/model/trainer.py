import xgboost as xgb
from sklearn.model_selection import train_test_split
from config import OUTPUT_MODEL_PATH, TRAINING_CONFIG
import pandas as pd
from pathlib import Path

def train_xgboost(X, y, xgb_params):
    """
    Train an XGBoost classifier on the provided dataset.

    Parameters:
    - X: Feature matrix
    - y: Target labels
    - xgb_params: Dictionary of XGBoost hyperparameters

    Returns:
    - model: Trained XGBoost classifier
    - X_test: Test feature set
    """
    test_size = TRAINING_CONFIG.get("test_size", 0.2)
    random_state = TRAINING_CONFIG.get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    return model, X_test

# def train_xgboost_model(features_path, labels_path, output_model_path=OUTPUT_MODEL_PATH):
#     """
#     Train an XGBoost model from CSV feature and label files and save the model.

#     Parameters:
#     - features_path: Path to the CSV file containing feature data
#     - labels_path: Path to the CSV file containing target labels
#     - output_model_path: Path to save the trained model (default from config)

#     Returns:
#     - model: Trained XGBoost classifier
#     """
#     df_features = pd.read_csv(features_path)
#     df_labels = pd.read_csv(labels_path)

#     X = df_features
#     y = df_labels["risk_label"]

#     model = xgb.XGBClassifier(
#         n_estimators=100,
#         max_depth=5,
#         learning_rate=0.1,
#         use_label_encoder=False,
#         eval_metric='logloss'
#     )

#     model.fit(X, y)

#     Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
#     model.save_model(output_model_path)
#     print(f"✅ Trained XGBoost classifier saved to '{output_model_path}'")

#     return model
