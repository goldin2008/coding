# import pandas as pd
# from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, X_test

# def train_xgboost_model(features_path, labels_path, output_model_path="models/xgb_model.json"):
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
