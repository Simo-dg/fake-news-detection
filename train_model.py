import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data" / "balanced_dataset_200k.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODEL_DIR / "tfidf_logreg_improved.joblib"

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print(f"🔄 Loading data from {DATA_PATH}...")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: File not found at {DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    
    # Check columns (Adjust these names if your parquet file is different)
    # We look for 'text' (content) and 'label' (0=Real, 1=Fake)
    if 'text' not in df.columns:
        print("❌ Error: Dataset must have a 'text' column.")
        print(f"Found columns: {df.columns}")
        return
        
    # Attempt to find label column
    target_col = 'label'
    if 'label' not in df.columns:
        # Fallback: try to find a likely target column
        possible = [c for c in df.columns if 'label' in c or 'target' in c or 'fake' in c]
        if possible:
            target_col = possible[0]
            print(f"⚠️ 'label' column not found. Using '{target_col}' as target.")
        else:
            print("❌ Error: Could not identify a label/target column.")
            return

    # 2. Preprocessing
    print("🧹 Preprocessing...")
    # Drop empty rows
    df = df.dropna(subset=['text', target_col])
    # Ensure text is string
    df['text'] = df['text'].astype(str)
    
    X = df['text']
    y = df[target_col]

    # 3. Split Data
    print(f"✂️ Splitting {len(df)} rows into Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Define Pipeline
    # NOTE: We use named steps 'tfidf' and 'clf' so Interpret-Text can find them easily later.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            max_features=50000,    # Limit vocabulary size to keep model fast
            ngram_range=(1, 2),    # Capture bigrams (e.g. "breaking news")
            lowercase=True
        )),
        ('clf', LogisticRegression(
            solver='liblinear',    # Good for high-dimensional text data
            C=1.0, 
            random_state=42,
            n_jobs=-1              # Use all CPU cores
        ))
    ])

    # 5. Train
    print("🚀 Training model (this might take a minute)...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("📊 Evaluating...")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n✅ Model Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 7. Save
    print(f"💾 Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print("🎉 Done! You can now run the Streamlit app.")

if __name__ == "__main__":
    train()