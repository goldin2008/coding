import pandas as pd

def load_feature_descriptions(path="data/generated/feature_library.csv", feature_names=None):
    try:
        df = pd.read_csv(path)
        return df.set_index("Feature Name")["Feature Meaning"].to_dict()
    except FileNotFoundError:
        return {f: f for f in feature_names}  # fallback
