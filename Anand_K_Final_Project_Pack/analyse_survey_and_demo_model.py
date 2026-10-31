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
from datetime import datetime
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
        responses = df[col].dropna()
        counts = responses.value_counts()
        for value, count in counts.items():
            tables.append({"question": col, "response": str(value), "frequency": int(count),
                           "percentage": round(100 * count / len(responses), 1)})
    pd.DataFrame(tables).to_csv(out_dir / "Survey_Frequency_Tables.csv", index=False)

    rating_question_numbers = [9, 10, 11, 14, 15]
    distribution_names = {
        9: "satisfaction",
        10: "reliability",
        11: "value",
        14: "continue",
        15: "switch",
    }
    rating_cols = [(number, df.columns[number]) for number in rating_question_numbers]
    summary = []
    for question_number, col in rating_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        answered_count = int(values.notna().sum())
        ratings_4_or_5_count = int((values >= 4).sum())
        summary.append({
            "measure": col, "mean_1_to_5": round(values.mean(), 3),
            "ratings_4_or_5_n": ratings_4_or_5_count,
            "ratings_4_or_5_pct": round(100 * ratings_4_or_5_count / answered_count, 1)
            if answered_count else float("nan")
        })
        counts = values.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        plt.figure(figsize=(7, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("Rating")
        plt.ylabel("Number of responses")
        plt.title(col.split(". ", 1)[-1])
        plt.tight_layout()
        plt.savefig(out_dir / f"rating_{question_number}.png", dpi=180)
        plt.savefig(out_dir / f"{distribution_names[question_number]}_distribution.png", dpi=180)
        plt.close()
    pd.DataFrame(summary).to_csv(out_dir / "Survey_Rating_Summary.csv", index=False)
    print("\nRating summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nInterpretation reminder: switching intention is not actual churn.")

def survey_switch_intention_demo(path, out_dir):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)

    df = pd.read_csv(path)
    switch_col = next((col for col in df.columns if col.startswith("15.")), None)
    if switch_col is None:
        raise ValueError("Expected survey question 15 for switching intention.")

    feature_cols = list(df.columns[1:14])
    switch_rating = pd.to_numeric(df[switch_col], errors="coerce")
    answered = switch_rating.notna()
    X = df.loc[answered, feature_cols]
    y = (switch_rating.loc[answered] >= 4).astype(int)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "Need at least two answered survey responses in each switching-intention class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    models = {
        "Majority-class baseline (always 1-3)": DummyClassifier(strategy="most_frequent"),
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
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": name,
            "target": "High stated switching intention (rating 4-5)",
            "test_samples": len(y_test),
            "test_high_intention_cases": int(y_test.sum()),
            "test_other_cases": int((y_test == 0).sum()),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        })
        print(f"\n{name} confusion matrix (rows=actual 1-3/4-5, columns=predicted 1-3/4-5):\n"
              f"{confusion_matrix(y_test, pred, labels=[0, 1])}")

    out = pd.DataFrame(rows)
    result_path = out_dir / "Survey_Switch_Intention_Model_DEMO_ONLY.csv"
    try:
        out.to_csv(result_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = result_path.with_name(
            f"{result_path.stem}_{timestamp}{result_path.suffix}"
        )
        out.to_csv(fallback_path, index=False)
        print(
            f"\nCould not write {result_path}; it may be open in another program "
            f"or protected. Results saved to {fallback_path} instead."
        )
    print("\nSURVEY SWITCHING-INTENTION DEMO ONLY — this predicts stated intent, not actual churn:")
    print(out.to_string(index=False))

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
    result_path = out_dir / "Synthetic_Model_Comparison_DEMO_ONLY.csv"
    try:
        out.to_csv(result_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = result_path.with_name(
            f"{result_path.stem}_{timestamp}{result_path.suffix}"
        )
        out.to_csv(fallback_path, index=False)
        print(
            f"\nCould not write {result_path}; it may be open in another program "
            f"or protected. Results saved to {fallback_path} instead."
        )
    print("\nSIMULATION ONLY — not evidence of actual customer churn performance:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="CSV containing timestamp and 17 questionnaire items")
    parser.add_argument("--survey-switch-demo", action="store_true",
                        help="Train demo models to predict stated switching intention from survey questions 1-13")
    parser.add_argument("--synthetic-model", help="Optional separate synthetic churn dataset")
    parser.add_argument("--out", default="analysis_outputs")
    args = parser.parse_args()
    out_dir = Path(args.out)
    survey_analysis(Path(args.survey), out_dir)
    if args.survey_switch_demo:
        survey_switch_intention_demo(Path(args.survey), out_dir)
    if args.synthetic_model:
        synthetic_model_demo(Path(args.synthetic_model), out_dir)
