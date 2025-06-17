import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

import sys
import logging
from sklearn.datasets import make_classification


# Import from project root
sys.path.append(str(Path(__file__).parent.parent))
from config.config import TRAINING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    logger.info(f"Splitting data (test_size={test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info("Training XGBoost model...")
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

if __name__ == "__main__":
    """Test the training functionality with synthetic data"""
    print("\n=== Testing XGBoost Training ===")
    
    # Test 1: Direct training with numpy arrays
    print("\nTest 1: Training with synthetic data...")
    X, y = make_classification(
        n_samples=100, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    
    xgb_params = {
        "n_estimators": 10,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }
    model, X_test = train_xgboost(X, y, xgb_params)
    print(f"✅ Success! Model trained. Test set size: {len(X_test)}")
    print(f"Feature importances: {model.feature_importances_[:5]}...")