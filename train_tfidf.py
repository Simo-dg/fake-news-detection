# train_tfidf.py
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)
PLOTS = BASE / "plots";  PLOTS.mkdir(exist_ok=True)

def main():
    import pandas as pd
    df = pd.read_parquet(DATA/"balanced_dataset_200k.parquet")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.label, test_size=0.2,
                                              stratify=df.label, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:,1]

    report = classification_report(y_te, y_pred, target_names=["REAL","FAKE"])
    (BASE/"evaluation_tfidf.txt").write_text(report)
    print(report)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title("Confusion Matrix — TF-IDF + LogReg")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(PLOTS/"tfidf_confusion.png", dpi=180); plt.close()

    fpr,tpr,_ = roc_curve(y_te, y_proba); roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.title("ROC — TF-IDF + LogReg"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"tfidf_roc.png", dpi=180); plt.close()

    joblib.dump(pipe, MODELS/"tfidf_logreg.joblib")
    print("Saved:", MODELS/"tfidf_logreg.joblib")

if __name__ == "__main__":
    main()
