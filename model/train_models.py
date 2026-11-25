from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

seed = 42
out_dir = Path(__file__).resolve().parent

# Match categorical column names regardless of spaces vs underscores
wanted_cats = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def norm_name(col):
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def make_prep(num_cols, categorical_cols):
    # Scale numeric columns and encode categorical columns inside pipelines
    num_flow = Pipeline([
        ("fill_na", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_flow = Pipeline([
        ("fill_na", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_flow, num_cols),
        ("cat", cat_flow, categorical_cols),
    ])


def get_scores(model_pipe, X_val, y_val):
    pred_y = model_pipe.predict(X_val)
    pred_probs = model_pipe.predict_proba(X_val)
    return {
        "Accuracy": round(float(accuracy_score(y_val, pred_y)), 4),
        "AUC": round(float(roc_auc_score(y_val, pred_probs, multi_class="ovr", average="weighted")), 4),
        "Precision": round(float(precision_score(y_val, pred_y, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_val, pred_y, average="weighted", zero_division=0)), 4),
        "F1": round(float(f1_score(y_val, pred_y, average="weighted", zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_val, pred_y)), 4),
    }


def main():
    uci = fetch_ucirepo(id=697)
    X = uci.data.features.copy()
    y_raw = uci.data.targets.squeeze()
    X.columns = [c.strip() for c in X.columns]

    df = pd.concat([X, y_raw.rename("Target")], axis=1)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Target"])
    le = LabelEncoder()
    y = le.fit_transform(df["Target"])

    # Map normalized names back to the real column names in X
    norm_to_col = {norm_name(c): c for c in X.columns}
    cat_cols = [norm_to_col[norm_name(w)] for w in wanted_cats if norm_name(w) in norm_to_col]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Same 80:20 stratified split so test set stays untouched during preprocessing fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    prep = make_prep(num_cols, cat_cols)

    # All 5 required classifiers on the same dataset and split
    all_models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed), "logreg.joblib"),
        "Decision Tree": (DecisionTreeClassifier(max_depth=10, random_state=seed), "dtree.joblib"),
        "kNN": (KNeighborsClassifier(n_neighbors=7), "knn.joblib"),
        "Naive Bayes": (GaussianNB(), "nb.joblib"),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=seed), "rf.joblib"),
    }

    rows = []
    for model_label, (clf, save_name) in all_models.items():
        # Fitting inside Pipeline keeps preprocessing fitted strictly on X_train
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out_dir / save_name)
        scores = get_scores(pipe, X_test, y_test)
        rows.append({"ML Model Name": model_label, **scores})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()