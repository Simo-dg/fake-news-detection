# train_tfidf_improved.py
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
from utils_data import load_true_fake

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)
PLOTS = BASE / "plots";  PLOTS.mkdir(exist_ok=True)

def main():
    df = load_true_fake(DATA/"True.csv", DATA/"Fake.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.label, test_size=0.2,
                                              stratify=df.label, random_state=42)

    # Improved TF-IDF with preprocessing
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),        # unigrams + bigrams
            min_df=5,                   # ignore very rare terms
            max_df=0.7,                 # ignore very common terms (more aggressive)
            max_features=50000,         # limit vocabulary size
            lowercase=True,             # CRITICAL: normalize case (US -> us)
            stop_words='english',       # remove common words like "this", "is"
            strip_accents='unicode',    # normalize accents
            sublinear_tf=True          # use log scaling for term frequency
        )),
        ("clf", LogisticRegression(
            max_iter=1000, 
            n_jobs=-1,
            class_weight='balanced',    # handle any class imbalance
            C=1.0                       # regularization strength
        ))
    ])
    
    print("Training improved TF-IDF model...")
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:,1]

    report = classification_report(y_te, y_pred, target_names=["REAL","FAKE"])
    (BASE/"evaluation_tfidf_improved.txt").write_text(report)
    print("\n" + "="*60)
    print("IMPROVED TF-IDF RESULTS")
    print("="*60)
    print(report)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title("Confusion Matrix — Improved TF-IDF + LogReg")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(PLOTS/"tfidf_improved_confusion.png", dpi=180); plt.close()

    fpr,tpr,_ = roc_curve(y_te, y_proba); roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.title("ROC — Improved TF-IDF + LogReg"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"tfidf_improved_roc.png", dpi=180); plt.close()

    joblib.dump(pipe, MODELS/"tfidf_logreg_improved.joblib")
    print(f"\nSaved: {MODELS/'tfidf_logreg_improved.joblib'}")
    
    # Show top features for interpretation
    print("\n" + "="*60)
    print("TOP FEATURES ANALYSIS")
    print("="*60)
    
    vectorizer = pipe.named_steps['tfidf']
    logreg = pipe.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = logreg.coef_[0]
    
    # Top features pushing toward FAKE (positive coefficients)
    top_fake_idx = np.argsort(coefs)[-20:][::-1]
    print("\nTop 20 features indicating FAKE news:")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:30s} : {coefs[idx]:+.4f}")
    
    # Top features pushing toward REAL (negative coefficients)
    top_real_idx = np.argsort(coefs)[:20]
    print("\nTop 20 features indicating REAL news:")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:30s} : {coefs[idx]:+.4f}")

if __name__ == "__main__":
    main()
