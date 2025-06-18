import pandas as pd
from config import DATA_PATHS

def load_feature_descriptions(path=DATA_PATHS["feature_library_csv"], feature_names=None):
    """
    Loads feature descriptions from a CSV file. If the file is not found,
    it falls back to returning a dictionary with feature names mapping to themselves.
    """
    try:
        df = pd.read_csv(path)
        return df.set_index("Feature Name")["Feature Meaning"].to_dict()
    except FileNotFoundError:
        return {f: f for f in feature_names}  # fallback


# feature_library.py

feature_library = {
    "wirein_ct": "Number of incoming wire transactions. A high value may indicate frequent transfers that could mask suspicious activity.",
    "wireout_ct": "Number of outgoing wire transactions. Frequent outgoing wires might indicate rapid movement of funds.",
    "avg_wire_amount": "Average amount per wire transaction. Large average amounts can indicate high-value transfers.",
    "max_wire_amount": "Maximum amount among all wire transactions. Large maximum values might flag high-risk, one-off transactions.",
    "perc_wire_to_high_risk_country": "Percentage of wire transactions sent to high-risk countries. Higher percentages increase money laundering risk.",
    "perc_wire_from_high_risk_country": "Percentage of wire transactions received from high-risk countries. Incoming transactions from such countries are suspicious.",
    "num_high_risk_counterparties": "Number of counterparties flagged as high risk. More counterparties can increase exposure.",
    "same_day_wire_ratio": "Ratio of same-day wire transactions to total transactions. Rapid funds movement is a red flag.",
    "suspicious_counterparty_score": "Score representing the risk level of counterparties based on known suspicious behavior.",
    "txn_frequency": "Frequency of transactions over a defined period. Unusually high frequency may indicate layering.",
    "degree_centrality": "Network centrality measure indicating how connected an entity is in the network of transactions.",
    "betweenness_centrality": "Network centrality indicating the extent to which an entity connects different parts of the network.",
    "clustering_coefficient": "Network clustering measure indicating if transactions are tightly grouped. High clustering can hide suspicious relationships.",
    "cash_deposit_amt": "Total amount of cash deposits. Large cash deposits may flag structuring or placement risk.",
    "cash_withdrawal_amt": "Total amount of cash withdrawals. Large withdrawals may also signal suspicious movement of funds.",
    "kyc_flag": "Flag indicating incomplete or outdated Know Your Customer documentation.",
    "num_sar_filings": "Number of suspicious activity reports filed for the entity. A history of SAR filings is a significant risk indicator.",
    "past_sar_flag": "Whether there was at least one SAR filed in the past. Indicates known suspicious activity.",
    "customer_age": "Age of the customer. Younger or older customers may have different risk profiles.",
    "account_tenure": "How long the account has been open. Newer accounts might pose higher risk.",
    "txn_amount_std": "Standard deviation of transaction amounts. High variability might indicate inconsistent or suspicious behavior.",
    "num_large_cash_txns": "Number of large cash transactions. Large cash transactions may flag structuring or placement.",
    "international_txn_ratio": "Ratio of international transactions to total transactions. High ratios can increase cross-border risk.",
    "num_unusual_txn_patterns": "Number of transactions that deviate from typical patterns. Anomalies are often suspicious.",
    "num_new_accounts_opened": "Number of new accounts opened. Rapid account opening can indicate attempts to evade monitoring.",
    "recent_address_change_flag": "Indicates if the customer recently changed address. Frequent changes might hide suspicious identity.",
    "pep_flag": "Politically exposed person flag. PEPs are higher risk due to potential corruption and bribery.",
    "num_credit_card_txns": "Number of credit card transactions. High usage might mask other illicit activities.",
    "num_failed_login_attempts": "Number of failed login attempts. Could indicate account takeover or identity theft.",
    "recent_device_change_flag": "Indicates if the customer recently changed devices. Device hopping might avoid detection."
}