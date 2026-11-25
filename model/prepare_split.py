import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

seed = 42

# Pulling UCI dataset 697 (Predict Student's Dropout and Academic Success)
# Has student demographics, academic background, economics, and sem 1/2 grades
raw_uci = fetch_ucirepo(id=697)
feat_df = raw_uci.data.features.copy()
target_series = raw_uci.data.targets.squeeze()

# Strips stray spaces/tabs from column names so lookups don't fail later
feat_df.columns = [c.strip() for c in feat_df.columns]

full_df = pd.concat([feat_df, target_series.rename("Target")], axis=1)

# Drops exact duplicate rows if any exist before splitting
full_df = full_df.drop_duplicates().reset_index(drop=True)

print("Dataset shape:", full_df.shape)
print("Missing values:", full_df.isna().sum().sum())

# Checks class counts (Graduate, Enrolled, Dropout) to see imbalance across target classes
print("\nTarget class counts:")
print(full_df["Target"].value_counts())

# 80/20 stratified split so the test set keeps the same class ratios as the full data
_, X_holdout, _, y_holdout = train_test_split(
    full_df.drop(columns=["Target"]),
    full_df["Target"],
    test_size=0.2,
    stratify=full_df["Target"],
    random_state=seed,
)

holdout_df = X_holdout.copy()
holdout_df["Target"] = y_holdout.values
holdout_df.to_csv("test_data.csv", index=False)
print("\nSaved test_data.csv:", holdout_df.shape)