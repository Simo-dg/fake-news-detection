# evaluate_models.py (fast)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from utils_data import load_true_fake

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"
PLOTS = BASE / "plots"; PLOTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BATCH = 64 if torch.cuda.is_available() else 32
MAX_LEN = 256  # speed-up a lot vs 512

def plot_roc(y, proba_dict, out):
    plt.figure(figsize=(6,5))
    for name, p1 in proba_dict.items():
        fpr, tpr, _ = roc_curve(y, p1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.title("ROC — Models (test set)")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def plot_cm(y, pred, name, out):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title(f"Confusion — {name} (test)"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def infer_finetuned(texts):
    tok = AutoTokenizer.from_pretrained(MODELS/"bert_finetuned")
    mdl = AutoModelForSequenceClassification.from_pretrained(MODELS/"bert_finetuned")
    mdl.to(DEVICE, dtype=DTYPE).eval()
    probs, preds = [], []
    with torch.inference_mode():
        for batch in iter_batches(texts, BATCH):
            enc = tok(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            out = mdl(**enc).logits.softmax(-1)
            probs.append(out[:,1].detach().cpu().numpy())
            preds.append(out.argmax(-1).detach().cpu().numpy())
    p1 = np.concatenate(probs); pr = np.concatenate(preds)
    return pr, p1

def embed_frozen(texts, hf_model, cache_path):
    # cache to avoid recomputing
    if cache_path.exists():
        return np.load(cache_path)
    tok = AutoTokenizer.from_pretrained(hf_model)
    enc_m = AutoModel.from_pretrained(hf_model)
    enc_m.to(DEVICE, dtype=DTYPE).eval()
    embs = []
    with torch.inference_mode():
        for batch in iter_batches(texts, BATCH):
            enc = tok(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            out = enc_m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.detach().cpu().numpy())
    E = np.vstack(embs)
    np.save(cache_path, E)
    return E

def main():
    # load and split ONCE; evaluate on TEST ONLY
    df = load_true_fake(DATA/"True.csv", DATA/"Fake.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    X = test_df["text"].tolist()
    y = test_df["label"].to_numpy()

    preds, probas = {}, {}

    # TF-IDF
    tfidf_path = MODELS/"tfidf_logreg.joblib"
    if tfidf_path.exists():
        pipe = joblib.load(tfidf_path)
        p = pipe.predict_proba(X)[:,1]
        pr = (p >= 0.5).astype(int)
        preds["TF-IDF"], probas["TF-IDF"] = pr, p
        plot_cm(y, pr, "TF-IDF", PLOTS/"cm_tfidf.png")

    # BERT (frozen) + LogReg
    base_cfg = MODELS/"bert_base_config.json"
    base_clf = MODELS/"bert_base_logreg.joblib"
    if base_cfg.exists() and base_clf.exists():
        cfg = json.loads(base_cfg.read_text())
        clf = joblib.load(base_clf)
        cache = MODELS/"bert_base_test_embeds.npy"
        E = embed_frozen(X, cfg["hf_model"], cache)
        p = clf.predict_proba(E)[:,1]
        pr = (p >= 0.5).astype(int)
        preds["BERT base"], probas["BERT base"] = pr, p
        plot_cm(y, pr, "BERT base", PLOTS/"cm_bert_base.png")

    # BERT fine-tuned
    if (MODELS/"bert_finetuned").exists():
        pr, p = infer_finetuned(X)
        preds["BERT fine-tuned"], probas["BERT fine-tuned"] = pr, p
        plot_cm(y, pr, "BERT fine-tuned", PLOTS/"cm_bert_ft.png")

    # Combined ROC
    if probas:
        plot_roc(y, probas, PLOTS/"compare_roc_test_only.png")

    print("Done. Plots in", PLOTS)

if __name__ == "__main__":
    main()
