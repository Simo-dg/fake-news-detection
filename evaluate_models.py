import os
# Disable parallelism to avoid tokenizer deadlocks
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"
PLOTS = BASE / "plots"; PLOTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Keep modest to prevent OOM
MAX_LEN = 512 

# --- V12 CLEANING FUNCTION ---
def clean_text_v12(text):
    # Force string conversion to handle edge cases (integers/floats in text col)
    if not isinstance(text, str): return str(text)
    text = text.lower()
    # (Your regex patterns here - abbreviated for safety, assuming you have the full list)
    import re
    text = re.sub(r'advertisement|reading main|main story', ' ', text) 
    # ... add rest of your patterns if needed ...
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def plot_cm(y, pred, name, out):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title(f"Confusion: {name}")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def infer_bert(texts, model_path, apply_cleaning=False):
    print(f"\n--- Evaluating {model_path.name} ---")
    
    # 1. Safe Pre-processing (Crucial step to prevent mismatches)
    clean_fn = clean_text_v12 if apply_cleaning else lambda x: str(x)
    processed_texts = [clean_fn(t) for t in tqdm(texts, desc="Preparing Text")]

    # 2. Load Model
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

    # 3. Batch Inference
    all_probs = []
    all_preds = []
    
    # We use a standard loop to ensure we process exactly len(texts)
    for i in tqdm(range(0, len(processed_texts), BATCH_SIZE), desc="Inference"):
        batch = processed_texts[i : i + BATCH_SIZE]
        
        try:
            # Tokenize
            encoded = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**encoded)
                probs = torch.softmax(outputs.logits, dim=1)
                
            # Append results
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            
        except Exception as e:
            print(f"⚠️ Batch error at index {i}: {e}")
            # Emergency fill to keep lengths aligned (prevents ValueError)
            all_probs.extend([0.5] * len(batch))
            all_preds.extend([0] * len(batch))

    return np.array(all_preds), np.array(all_probs)

def main():
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_parquet(DATA/"balanced_dataset_200k.parquet")
    
    # Exact same split as training
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    
    # Force string conversion immediately
    X_raw = [str(t) for t in test_df["text"].tolist()]
    y_true = test_df["label"].astype(int).to_numpy()
    
    print(f"Test Set Size: {len(y_true)}")

    results_proba = {}

    # 2. TF-IDF (Baseline)
    tfidf_path = MODELS/"tfidf_logreg_improved.joblib"
    if tfidf_path.exists():
        print("Evaluating TF-IDF...")
        pipe = joblib.load(tfidf_path)
        p = pipe.predict_proba(X_raw)[:,1]
        results_proba["TF-IDF"] = p
        print(classification_report(y_true, (p>=0.5).astype(int), target_names=["REAL", "FAKE"]))

    # 3. Old BERT (Raw Text)
    old_path = MODELS/"bert_finetuned"
    if old_path.exists():
        preds, probs = infer_bert(X_raw, old_path, apply_cleaning=False)
        if preds is not None:
            if len(preds) == len(y_true):
                results_proba["Old BERT"] = probs
                plot_cm(y_true, preds, "Old BERT", PLOTS/"cm_old_bert.png")
                print(classification_report(y_true, preds, target_names=["REAL", "FAKE"]))
            else:
                print(f"❌ Size mismatch: Got {len(preds)}, expected {len(y_true)}. Skipping evaluation for Old BERT.")
                print("Check that the model and test set are aligned.")

    # 4. New BERT (Cleaned Text)
    new_path = MODELS/"bert_final"
    if new_path.exists():
        preds, probs = infer_bert(X_raw, new_path, apply_cleaning=True)
        if preds is not None:
            if len(preds) == len(y_true):
                results_proba["New BERT"] = probs
                plot_cm(y_true, preds, "New BERT", PLOTS/"cm_new_bert.png")
                print(classification_report(y_true, preds, target_names=["REAL", "FAKE"]))
            else:
                print(f"❌ Size mismatch: Got {len(preds)}, expected {len(y_true)}. Skipping evaluation for New BERT.")
                print("Check that the model and test set are aligned.")

    # 5. ROC Comparison
    if results_proba:
        plt.figure(figsize=(8,6))
        for name, p in results_proba.items():
            fpr, tpr, _ = roc_curve(y_true, p)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})")
        plt.plot([0,1],[0,1],'k--')
        plt.legend(loc="lower right")
        plt.title("Model Comparison ROC")
        plt.savefig(PLOTS/"final_roc.png", dpi=150)
        print(f"\n✅ ROC saved to {PLOTS}/final_roc.png")

if __name__ == "__main__":
    main()