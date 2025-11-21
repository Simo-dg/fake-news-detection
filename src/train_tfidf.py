from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re # Added re for regex cleaning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import logging
from tqdm import tqdm
import pandas as pd
import config

DATA = config.DATA_DIR
MODELS = config.MODELS_DIR
PLOTS = config.PLOTS_DIR



# --- NEW CLEANING FUNCTION ---
def clean_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. NORMALIZE PUNCTUATION
    # Turn all punctuation into spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. FORBIDDEN WORDS (The "Atomic" List)
    # We remove these words entirely.
    forbidden_words = [
        r'\bnews\b', r'\bsource\b', r'\bcents\b', 
        r'\breaders\b',  # +12 artifact ("readers think")
        r'\badd\b',      # +8 artifact ("add comment")
        r'\bimg\b', r'\bpic\b'
    ]
    combined_forbidden = re.compile('|'.join(forbidden_words))
    text = combined_forbidden.sub(' ', text)

    # 4. PHRASE ARTIFACTS (The "Final Four" & Real News Leaks)
    patterns = [
        # The "Widget Fragments" you just found
        r'fact\s+add', 
        r'think\s+story', 
        r'story\s+fact', 
        r'fact\s+think',
        r'read\s+more', r'click\s+here', 
        r'sign\s+up', r'check\s+out',
        
        # Real News Scraper Leaks
        r'reading\s+main', r'main\s+story', r'continue\s+reading',
        r'president\s+elect', r'source\s+text', 
        r'rights\s+reserved', r'copyright', 
        r'pm\s+et',          # Found in V10 logs
        r'latest\s+video',   # Found in V10 logs
        r'lat\s+video'
    ]
    combined_patterns = re.compile('|'.join(patterns))
    text = combined_patterns.sub(' ', text)

    # 5. EDITORIAL / CREDITS
    meta_words = [
        r'\bphoto\b', r'\bimage\b', r'\bcredit\b', 
        r'\beditor\b', r'\bediting\b', 
        r'\bwriter\b', r'\bphotograph\b', 
        r'\bcolumn\b', r'\beditorial\b',
        r'\bcaption\b', r'\breporting\b'
    ]
    for p in meta_words:
        text = re.sub(p, ' ', text)

    # 6. AGENCIES
    agencies = [
        r'\b(reuters|ap|afp|upi|bloomberg|cnbc|cnn|bbc|nyt|new york times|washington post)\b',
        r'\b(nytimes|breitbart|christian post|consortiumnews|daily caller)\b',
        r'\b(calif|gmt|est|pst)\b'
    ]
    for pattern in agencies:
        text = re.sub(pattern, ' ', text)

    # 7. TEMPORAL
    months = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b'
    text = re.sub(months, ' ', text)
    
    # Final Cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    # 1. LOAD
    logging.info("version 11")
    logging.info("Uploading balanced dataset of 200k...")
    df = pd.read_parquet(DATA/"balanced_dataset_200k.parquet")
    logging.info(f"Total raw articles: {len(df):,}")
    
    # 2. CLEAN (This is the step that was missing!)
    logging.info("Applying aggressive artifact cleaning (removing 'Reuters', 'Featured Image', etc)...")
    # Using tqdm for progress bar since 200k rows takes a moment
    tqdm.pandas() 
    df['text'] = df['text'].progress_apply(clean_text)
    
    # Remove empty rows after cleaning
    df = df[df['text'].str.len() > 50] 
    logging.info(f"Articles remaining after cleaning: {len(df):,}")

    # 3. SPLIT
    logging.info("Splitting train/test...")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.label, test_size=0.2,
                                              stratify=df.label, random_state=42)
    logging.info(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")

    # 4. PIPELINE
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=10,                  # Increased to reduce noise
            max_df=0.5,                 # Decreased: Ignore words that appear in >50% of docs
            max_features=50000,
            lowercase=True,
            stop_words='english',
            strip_accents='unicode',
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000, 
            n_jobs=-1,

            C=1.0
        ))
    ])

    logging.info("Starting training of ROBUST TF-IDF model...")
    pipe.fit(X_tr, y_tr)
    logging.info("Training completed.")

    # 5. EVALUATE
    logging.info("Predicting on test set...")
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:,1]

    logging.info("Evaluating...")
    report = classification_report(y_te, y_pred, target_names=["REAL","FAKE"])
    (MODELS/"evaluation_tfidf_robust.txt").write_text(report)
    print("\n" + "="*60)
    print("ROBUST TF-IDF RESULTS")
    print("="*60)
    print(report)

    # 6. PLOT
    logging.info("Saving plots...")
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title("Confusion Matrix — Robust TF-IDF")
    plt.tight_layout(); plt.savefig(PLOTS/"tfidf_robust_confusion.png", dpi=180); plt.close()

    fpr,tpr,_ = roc_curve(y_te, y_proba); roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.title("ROC — Robust TF-IDF"); plt.legend(); plt.savefig(PLOTS/"tfidf_robust_roc.png", dpi=180); plt.close()

    # 7. FEATURE ANALYSIS
    logging.info("Analyzing top features...")
    vectorizer = pipe.named_steps['tfidf']
    logreg = pipe.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = logreg.coef_[0]
    
    top_fake_idx = np.argsort(coefs)[-20:][::-1]
    print("\nTop 20 features indicating FAKE news (Should NOT be 'featured image'):")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:30s} : {coefs[idx]:+.4f}")
        
    top_real_idx = np.argsort(coefs)[:20]
    print("\nTop 20 features indicating REAL news (Should NOT be 'reuters'):")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:30s} : {coefs[idx]:+.4f}")

    # 8. SAVE MODEL
    logging.info("Saving robust TF-IDF model to disk...")
    joblib.dump(pipe, MODELS/"tfidf_logreg_robust.joblib")
    logging.info(f"Model saved: {MODELS/'tfidf_logreg_robust.joblib'}")

if __name__ == "__main__":
    main()