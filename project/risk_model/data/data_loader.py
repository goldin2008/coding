import pandas as pd
import numpy as np
import os
from data.feature_library import feature_library
from config import (
    DATA_PATHS,
    DATA_GENERATION,
    TRAINING_CONFIG,
)

FEATURE_LIBRARY = feature_library

def generate_feature_library_csv(output_path=DATA_PATHS["feature_library_csv"]):
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
    num_samples=DATA_GENERATION["num_samples"], 
    fraud_ratio=DATA_GENERATION["fraud_ratio"], 
    output_path=DATA_PATHS["aml_data"], 
    random_state=TRAINING_CONFIG["random_state"]
):
    np.random.seed(random_state)

    n_fraud = int(num_samples * fraud_ratio)
    n_normal = num_samples - n_fraud

    data = {
        "wirein_ct": np.random.poisson(3, num_samples),
        "wireout_ct": np.random.poisson(3, num_samples),
        "avg_wire_amount": np.random.uniform(500, 5000, num_samples),
        "max_wire_amount": np.random.uniform(1000, 10000, num_samples),
        "perc_wire_to_high_risk_country": np.random.uniform(0, 1, num_samples),
        "perc_wire_from_high_risk_country": np.random.uniform(0, 1, num_samples),
        "num_high_risk_counterparties": np.random.poisson(2, num_samples),
        "same_day_wire_ratio": np.random.uniform(0, 0.5, num_samples),
        "suspicious_counterparty_score": np.random.uniform(0, 1, num_samples),
        "txn_frequency": np.random.randint(1, 20, num_samples),
        "degree_centrality": np.random.uniform(0, 1, num_samples),
        "betweenness_centrality": np.random.uniform(0, 1, num_samples),
        "clustering_coefficient": np.random.uniform(0, 1, num_samples),
        "cash_deposit_amt": np.random.uniform(1000, 10000, num_samples),
        "cash_withdrawal_amt": np.random.uniform(1000, 10000, num_samples),
        "kyc_flag": np.random.choice([0, 1], num_samples),
        "num_sar_filings": np.random.poisson(0.5, num_samples),
        "past_sar_flag": np.random.choice([0, 1], num_samples),
        "customer_age": np.random.randint(18, 80, num_samples),
        "account_tenure": np.random.uniform(0, 20, num_samples),
        "txn_amount_std": np.random.uniform(50, 500, num_samples),
        "num_large_cash_txns": np.random.poisson(1, num_samples),
        "international_txn_ratio": np.random.uniform(0, 1, num_samples),
        "num_unusual_txn_patterns": np.random.poisson(2, num_samples),
        "num_new_accounts_opened": np.random.poisson(0.5, num_samples),
        "recent_address_change_flag": np.random.choice([0, 1], num_samples),
        "pep_flag": np.random.choice([0, 1], num_samples),
        "num_credit_card_txns": np.random.poisson(5, num_samples),
        "num_failed_login_attempts": np.random.poisson(1, num_samples),
        "recent_device_change_flag": np.random.choice([0, 1], num_samples)
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

    df["risk_label"] = labels

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data generated and saved to {output_path}")

    return df


def load_dataset(path=DATA_PATHS["aml_data"]):
    """
    Loads a dataset from a CSV file and splits it into features (X) and label (y).
    """
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col != "risk_label"]
    X = df[feature_cols]
    y = df["risk_label"]
    print(f"✅ Dataset loaded from {path} with shape {df.shape}")
    return X, y


def load_feature_descriptions(path=DATA_PATHS["feature_library_csv"], feature_names=None):
    """
    Loads feature descriptions from a CSV, or falls back to the FEATURE_LIBRARY dict.
    """
    try:
        df = pd.read_csv(path)
        return df.set_index("Feature Name")["Feature Meaning"].to_dict()
    except FileNotFoundError:
        return {f: f for f in feature_names}  # fallback