import pandas as pd
import numpy as np
from pathlib import Path

import os

# ✅ Predefined AML feature library
FEATURE_LIBRARY = {
    "wirein_ct": "Number of inbound wire transactions",
    "perc_hrg_wire_amt": "% of wire amount to high-risk geographies",
    "degree_centrality": "Network connectivity score (number of connections in transaction network)",
    "txn_frequency": "Transaction frequency over last 30 days",
    "avg_wire_amt": "Average amount per wire transaction",
    "same_day_wire_ratio": "% of same-day wire transactions",
    "high_risk_country_flag": "Indicator if customer operates in high-risk country",
    "cash_deposit_amt": "Total cash deposit amount",
    "kyc_flag": "Missing or incomplete KYC documentation",
    "suspicious_activity_flag": "Flagged by transaction monitoring system",
    "txn_amount_std": "Standard deviation of transaction amounts",
    "suspicious_counterparty_score": "Counterparty risk score based on past suspicious reports"
}


def generate_feature_library_csv(output_path="data/feature_library.csv"):
    """
    Generates a feature library CSV with feature names and descriptions.
    """
    feature_names = list(FEATURE_LIBRARY.keys())
    feature_meanings = [FEATURE_LIBRARY[feature] for feature in feature_names]

    feature_library_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Feature Meaning": feature_meanings
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_library_df.to_csv(output_path, index=False)
    print(f"✅ Feature library CSV saved to {output_path}")


def generate_and_save_aml_data(
    num_samples=1000, 
    fraud_ratio=0.3, 
    output_path="data/aml_data.csv", 
    random_state=42
):
    np.random.seed(random_state)

    n_fraud = int(num_samples * fraud_ratio)
    n_normal = num_samples - n_fraud

    data = {
        "wirein_ct": np.random.poisson(3, num_samples),
        "perc_hrg_wire_amt": np.random.uniform(0, 1, num_samples),
        "degree_centrality": np.random.uniform(0, 1, num_samples),
        "txn_frequency": np.random.randint(1, 20, num_samples),
        "avg_wire_amt": np.random.uniform(500, 5000, num_samples),
        "same_day_wire_ratio": np.random.uniform(0, 0.5, num_samples),
        "high_risk_country_flag": np.random.choice([0, 1], num_samples),
        "cash_deposit_amt": np.random.uniform(1000, 10000, num_samples),
        "kyc_flag": np.random.choice([0, 1], num_samples),
        "suspicious_activity_flag": np.random.choice([0, 1], num_samples),
        "txn_amount_std": np.random.uniform(50, 500, num_samples),
        "suspicious_counterparty_score": np.random.uniform(0, 1, num_samples)
    }
    
    df = pd.DataFrame(data)

    labels = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_fraud, dtype=int)
    ])

    # Shuffle data and labels together
    shuffled_indices = np.random.permutation(num_samples)
    df = df.iloc[shuffled_indices].reset_index(drop=True)
    labels = labels[shuffled_indices]

    # Add label column to the same DataFrame
    df["risk_label"] = labels

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Data generated and saved to {output_path}")

    return df

def load_dataset(path):
    """
    Loads a dataset from a CSV file and splits it into features (X) and label (y).
    """
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col != "risk_label"]
    X = df[feature_cols]
    y = df["risk_label"]
    print(f"✅ Dataset loaded from {path} with shape {df.shape}")
    return X, y

# def generate_dummy_dataset(num_samples=100):
#     """
#     Generates a dummy AML dataset for training/testing with a binary risk label.
#     """
#     np.random.seed(42)
#     data = {
#         "wirein_ct": np.random.poisson(3, num_samples),
#         "perc_hrg_wire_amt": np.random.uniform(0, 1, num_samples),
#         "degree_centrality": np.random.uniform(0, 1, num_samples),
#         "txn_frequency": np.random.randint(1, 20, num_samples),
#         "avg_wire_amt": np.random.uniform(500, 5000, num_samples),
#         "same_day_wire_ratio": np.random.uniform(0, 0.5, num_samples),
#         "high_risk_country_flag": np.random.choice([0, 1], num_samples),
#         "cash_deposit_amt": np.random.uniform(1000, 10000, num_samples),
#         "kyc_flag": np.random.choice([0, 1], num_samples),
#         "suspicious_activity_flag": np.random.choice([0, 1], num_samples),
#         "txn_amount_std": np.random.uniform(50, 500, num_samples),
#         "suspicious_counterparty_score": np.random.uniform(0, 1, num_samples)
#     }

#     df = pd.DataFrame(data)

#     # Use a simple weighted sum of some features to generate risk probabilities
#     risk_score = (
#         0.3 * df["wirein_ct"] +
#         0.4 * df["perc_hrg_wire_amt"] +
#         0.2 * df["degree_centrality"] +
#         0.1 * df["same_day_wire_ratio"] +
#         0.3 * df["high_risk_country_flag"] +
#         np.random.normal(0, 0.05, num_samples)
#     )

#     # Convert to probability between 0 and 1
#     risk_probability = 1 / (1 + np.exp(-risk_score))

#     # Assign binary label based on threshold (e.g., 0.5)
#     # risk_label = (risk_probability > 0.5).astype(int)
#     risk_label = risk_probability
#     df["risk_label"] = risk_label

#     print(f"✅ Dummy dataset with {num_samples} samples and binary risk label generated.")
#     return df



# def generate_synthetic_data(n_samples=1000, n_features=20, random_state=42, save_to_csv=True, output_dir="data/generated"):
#     np.random.seed(random_state)
#     feature_names = [f"feature_{i+1}" for i in range(n_features)]
#     features = np.random.rand(n_samples, n_features)
    
#     # Binary classification label (0 or 1)
#     risk_labels = np.random.randint(0, 2, size=n_samples)
    
#     df_features = pd.DataFrame(features, columns=feature_names)
#     df_labels = pd.DataFrame({"risk_label": risk_labels})

#     if save_to_csv:
#         Path(output_dir).mkdir(parents=True, exist_ok=True)
#         df_features.to_csv(f"{output_dir}/features.csv", index=False)
#         df_labels.to_csv(f"{output_dir}/labels.csv", index=False)
#         print(f"✅ Saved features.csv and labels.csv to '{output_dir}'")

#     return df_features, df_labels


# def generate_feature_library_csv(X, output_path="data/generated"):
#     """
#     Generates a CSV file containing feature names and placeholder descriptions.

#     Args:
#         X (pd.DataFrame): DataFrame containing feature columns.
#         output_path (str): Path to save the CSV file.

#     Returns:
#         None
#     """
#     feature_names = X.columns.tolist()
#     feature_meanings = ["No description available" for _ in feature_names]

#     feature_library_df = pd.DataFrame({
#         "Feature Name": feature_names,
#         "Feature Meaning": feature_meanings
#     })

#     feature_library_df.to_csv(output_path, index=False)
#     print(f"✅ Feature library CSV saved to {output_path}")
