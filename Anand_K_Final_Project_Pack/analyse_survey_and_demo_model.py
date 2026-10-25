"""
AI-Driven Customer Churn Prediction and Retention Analytics
Reproducible survey analysis + optional, separate synthetic-model demonstration.

Important:
- Survey data measures stated perceptions and intentions, not verified actual churn.
- The optional modelling routine requires a separate synthetic dataset with the target
  actual_churn_simulated. Its metrics are demonstration-only, never real-world evidence.

Install:
    pip install pandas matplotlib scikit-learn

Run survey analysis:
    python analyse_survey_and_demo_model.py --survey Survey_Responses_426_Cleaned.csv

Optional synthetic demo:
    python analyse_survey_and_demo_model.py --survey Survey_Responses_426_Cleaned.csv \
        --synthetic-model Synthetic_Churn_Model_Demo_300_Not_Real_Customers.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def survey_analysis(path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    if len(df.columns) < 18:
        raise ValueError("Expected timestamp plus 17 questionnaire columns.")
    # Normalize a punctuation-only category variant for analysis consistency.
    complaint_col = df.columns[12]
    df[complaint_col] = df[complaint_col].replace({"2-3 times": "2–3 times"})
    print(f"Survey rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    cat_cols = list(df.columns[1:8]) + list(df.columns[11:13]) + list(df.columns[15:18])
    tables = []
    for col in cat_cols:
        counts = df[col].fillna("No response").value_counts(dropna=False)
        for value, count in counts.items():
            tables.append({"question": col, "response": str(value), "frequency": int(count),
                           "percentage": round(100 * count / len(df), 1)})
    pd.DataFrame(tables).to_csv(out_dir / "Survey_Frequency_Tables.csv", index=False)

    rating_cols = [df.columns[i] for i in [8, 9, 10, 13, 14]]
    summary = []
    for col in rating_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        summary.append({
            "measure": col, "mean_1_to_5": round(values.mean(), 3),
            "ratings_4_or_5_n": int((values >= 4).sum()),
            "ratings_4_or_5_pct": round(100 * (values >= 4).mean(), 1)
        })
        counts = values.value_counts().sort_index()
        plt.figure(figsize=(7, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("Rating")
        plt.ylabel("Number of responses")
        plt.title(col.split(". ", 1)[-1])
        plt.tight_layout()
        plt.savefig(out_dir / (f"rating_{df.columns.get_loc(col)+1}.png"), dpi=180)
        plt.close()
    pd.DataFrame(summary).to_csv(out_dir / "Survey_Rating_Summary.csv", index=False)
    print("\nRating summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nInterpretation reminder: switching intention is not actual churn.")

def synthetic_model_demo(path, out_dir):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    df = pd.read_csv(path)
    target = "actual_churn_simulated"
    if target not in df.columns:
        raise ValueError(f"Expected {target}; use only the separately labelled synthetic demo dataset.")
    y = df[target]
    X = df.drop(columns=[c for c in ["record_type", "customer_id", target] if c in df.columns])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6,
            min_samples_leaf=3, random_state=42, class_weight="balanced")
    }
    rows = []
    for name, clf in models.items():
        model = Pipeline([("preprocessor", prep), ("classifier", clf)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                     "precision": precision_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred, zero_division=0),
                     "f1": f1_score(y_test, pred, zero_division=0),
                     "roc_auc": roc_auc_score(y_test, proba)})
        print(f"\n{name} confusion matrix:\n{confusion_matrix(y_test, pred)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "Synthetic_Model_Comparison_DEMO_ONLY.csv", index=False)
    print("\nSIMULATION ONLY — not evidence of actual customer churn performance:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="CSV containing timestamp and 17 questionnaire items")
    parser.add_argument("--synthetic-model", help="Optional separate synthetic churn dataset")
    parser.add_argument("--out", default="analysis_outputs")
    args = parser.parse_args()
    out_dir = Path(args.out)
    survey_analysis(Path(args.survey), out_dir)
    if args.synthetic_model:
        synthetic_model_demo(Path(args.synthetic_model), out_dir)
