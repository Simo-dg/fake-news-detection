import os
# Disable parallelism to avoid tokenizer deadlocks
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report, 
    precision_recall_curve, 
    average_precision_score,
    accuracy_score,
    f1_score
)
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

# --- CONFIGURATION ---
DATA = config.DATA_DIR
MODELS = config.MODELS_DIR
PLOTS = config.PLOTS_DIR
TFIDF_PATH = config.TFIDF_PATH
BERT_OLD_PATH = config.BERT_OLD_PATH
BERT_FINAL_PATH = config.BERT_FINAL_PATH
DEVICE = config.DEVICE
BATCH_SIZE = config.EVAL_BATCH_SIZE
MAX_LEN = config.MAX_LEN

# --- STYLING ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
COLORS = {"TF-IDF": "#3498db", "Old BERT": "#95a5a6", "New BERT": "#e74c3c"}

# --- CLEANING FUNCTION ---
def clean_text_v12(text):
    if not isinstance(text, str): return str(text)
    text = text.lower()
    import re
    text = re.sub(r'advertisement|reading main|main story', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- INFERENCE ENGINE ---
def infer_bert(texts, model_path, apply_cleaning=False):
    print(f"\n--- Loading {model_path.name} ---")
    
    # Pre-processing
    clean_fn = clean_text_v12 if apply_cleaning else lambda x: str(x)
    processed_texts = [clean_fn(t) for t in tqdm(texts, desc="Processing Text")]

    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {e}")
        return None, None

    all_probs = []
    
    # Batch Inference
    for i in tqdm(range(0, len(processed_texts), BATCH_SIZE), desc="Inference"):
        batch = processed_texts[i : i + BATCH_SIZE]
        try:
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
                # We only need the probability of class 1 (FAKE)
                all_probs.extend(probs[:, 1].cpu().numpy())
            
        except Exception as e:
            print(f"⚠️ Error batch {i}: {e}")
            all_probs.extend([0.5] * len(batch)) # Neutral fill

    return np.array(all_probs)

# --- PLOTTING SUITE ---

def plot_confusion_matrix_enhanced(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black')
    
    # Overlay percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f"({cm_norm[i, j]:.1%})", 
                     ha="center", va="center", color="black", fontsize=10)
            
    plt.title(f"Confusion Matrix: {name}", fontsize=16, fontweight='bold')
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks([0.5, 1.5], ["REAL", "FAKE"])
    plt.yticks([0.5, 1.5], ["REAL", "FAKE"])
    plt.tight_layout()
    plt.savefig(PLOTS / f"cm_{name.replace(' ', '_').lower()}.png", dpi=200)
    plt.close()

def plot_calibration_curve_comparison(y_true, results):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for name, probs in results.items():
        if probs is None: continue
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name}", color=COLORS.get(name, "black"))
    
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Calibration Plot (Reliability Diagram)", fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "calibration_comparison.png", dpi=200)
    plt.close()

def plot_distributions(y_true, results):
    plt.figure(figsize=(10, 6))
    
    for name, probs in results.items():
        if probs is None: continue
        sns.kdeplot(probs, label=name, fill=True, alpha=0.3, linewidth=2, color=COLORS.get(name, "black"))
        
    plt.title("Probability Density Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Probability (0=Real, 1=Fake)")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "probability_density.png", dpi=200)
    plt.close()

def plot_roc_pr_comparison(y_true, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for name, probs in results.items():
        if probs is None: continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2, color=COLORS.get(name))
    
    ax1.set_title("ROC Curve", fontsize=14, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    
    # Precision-Recall
    for name, probs in results.items():
        if probs is None: continue
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax2.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", lw=2, color=COLORS.get(name))
        
    ax2.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(PLOTS / "roc_pr_comparison.png", dpi=200)
    plt.close()

def plot_model_correlations(results):
    # Create a DataFrame of probabilities
    df_probs = pd.DataFrame(results)
    
    plt.figure(figsize=(7, 6))
    # Spearman correlation (rank-based) is often better for probabilities
    corr = df_probs.corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", vmin=0.5, vmax=1.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title("Model Prediction Correlation (Spearman)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "model_correlation.png", dpi=200)
    plt.close()

def main():
    print("--- 1. Loading Data ---")
    df = pd.read_parquet(DATA/"balanced_dataset_200k.parquet")
    
    # Stratified Split (Same seed as training)
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    
    X_raw = [str(t) for t in test_df["text"].tolist()]
    y_true = test_df["label"].astype(int).to_numpy()
    
    print(f"Test Set Size: {len(y_true)}")
    results = {} # Stores Probabilities
    predictions = {} # Stores Binary Predictions (0/1)

    # --- 2. TF-IDF Evaluation ---
    if TFIDF_PATH.exists():
        print(f"\nEvaluating TF-IDF ({TFIDF_PATH.name})...")
        pipe = joblib.load(TFIDF_PATH)
        probs = pipe.predict_proba(X_raw)[:, 1]
        results["TF-IDF"] = probs
        predictions["TF-IDF"] = (probs >= 0.5).astype(int)
    else:
        print(f"⚠️ Skipping TF-IDF: {TFIDF_PATH} not found.")

    # --- 3. BERT Finetuned (Raw Text) ---
    if BERT_OLD_PATH.exists():
        probs = infer_bert(X_raw, BERT_OLD_PATH, apply_cleaning=False)
        if probs is not None:
            results["Old BERT"] = probs
            predictions["Old BERT"] = (probs >= 0.5).astype(int)
    else:
        print(f"⚠️ Skipping Old BERT: {BERT_OLD_PATH} not found.")

    # --- 4. BERT Final (Cleaned Text) ---
    if BERT_FINAL_PATH.exists():
        probs = infer_bert(X_raw, BERT_FINAL_PATH, apply_cleaning=True)
        if probs is not None:
            results["New BERT"] = probs
            predictions["New BERT"] = (probs >= 0.5).astype(int)
    else:
        print(f"⚠️ Skipping New BERT: {BERT_FINAL_PATH} not found.")

    # --- 5. Generate Reports & Plots ---
    if not results:
        print("❌ No models evaluated.")
        return

    print("\n--- Generating Visualizations ---")
    
    # A. Confusion Matrices
    for name, preds in predictions.items():
        plot_confusion_matrix_enhanced(y_true, preds, name)
        print(f"Saved CM for {name}")
        
        # Print Console Report
        print(f"\n--- Report: {name} ---")
        print(classification_report(y_true, preds, target_names=["REAL", "FAKE"], digits=4))

    # B. Comparative Plots
    plot_roc_pr_comparison(y_true, results)
    print("Saved ROC/PR Comparison")
    
    plot_distributions(y_true, results)
    print("Saved Probability Distributions")

    plot_calibration_curve_comparison(y_true, results)
    print("Saved Calibration Curves")
    
    if len(results) > 1:
        plot_model_correlations(results)
        print("Saved Correlation Heatmap")

    # C. Summary Metrics Table (Saved as CSV)
    summary_data = []
    for name, preds in predictions.items():
        probs = results[name]
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        auc_score = roc_curve(y_true, probs)[1] # Need full calc usually, but simplified here
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = auc(fpr, tpr)
        
        summary_data.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "AUC": auc_val
        })
    
    pd.DataFrame(summary_data).set_index("Model").to_csv(PLOTS / "final_metrics_summary.csv")
    print(f"\n✅ Done! All plots saved to: {PLOTS}")

if __name__ == "__main__":
    main()