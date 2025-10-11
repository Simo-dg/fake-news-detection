# train_bert_feature_extractor.py
from pathlib import Path
import json, joblib, numpy as np
from utils_data import load_true_fake
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib, matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)
PLOTS = BASE / "plots";  PLOTS.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

@torch.no_grad()
def embed_texts(texts, tok, mdl, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    mdl.to(device).eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, return_tensors="pt").to(device)
        out = mdl(**enc).last_hidden_state  # [B, T, H]
        mask = enc['attention_mask'].unsqueeze(-1)     # [B, T, 1]
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool -> [B, H]
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

def main():
    df = load_true_fake(DATA/"True.csv", DATA/"Fake.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text.tolist(), df.label.values,
                                              test_size=0.2, stratify=df.label, random_state=42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    for p in mdl.parameters(): p.requires_grad = False

    E_tr = embed_texts(X_tr, tok, mdl)
    E_te = embed_texts(X_te, tok, mdl)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(E_tr, y_tr)
    y_pred = clf.predict(E_te)
    y_proba = clf.predict_proba(E_te)[:,1]

    report = classification_report(y_te, y_pred, target_names=["REAL","FAKE"])
    (BASE/"evaluation_bert_base.txt").write_text(report)
    print(report)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    plt.title("Confusion Matrix — BERT (frozen) + LogReg")
    plt.tight_layout(); plt.savefig(PLOTS/"bert_base_confusion.png", dpi=180); plt.close()

    fpr,tpr,_ = roc_curve(y_te, y_proba); roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.title("ROC — BERT (frozen) + LogReg"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS/"bert_base_roc.png", dpi=180); plt.close()

    joblib.dump(clf, MODELS/"bert_base_logreg.joblib")
    (MODELS/"bert_base_config.json").write_text(json.dumps({"hf_model": MODEL_NAME}))
    print("Saved:", MODELS/"bert_base_logreg.joblib")

if __name__ == "__main__":
    main()
