# app.py
import json, os
from pathlib import Path
import numpy as np
import streamlit as st
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

BASE = Path(__file__).parent.resolve()
MODELS = BASE / "models"
PLOTS = BASE / "plots"

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector — Dashboard")

available = {
    "TF-IDF + LogReg": (MODELS/"tfidf_logreg.joblib").exists(),
    "BERT (frozen) + LogReg": (MODELS/"bert_base_logreg.joblib").exists() and (MODELS/"bert_base_config.json").exists(),
    "BERT (fine-tuned)": (MODELS/"bert_finetuned").exists()
}
choices = [k for k,v in available.items() if v]
if not choices:
    st.warning("Nessun modello trovato in /models. Addestra prima i modelli.")
    st.stop()

model_name = st.sidebar.selectbox("Seleziona modello", choices)

text = st.text_area("Incolla un articolo:", height=250, placeholder="Titolo e/o testo dell'articolo...")

@st.cache_resource
def load_tfidf():
    return joblib.load(MODELS/"tfidf_logreg.joblib")

@st.cache_resource
def load_bert_base():
    cfg = json.loads((MODELS/"bert_base_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["hf_model"])
    enc = AutoModel.from_pretrained(cfg["hf_model"])
    clf = joblib.load(MODELS/"bert_base_logreg.joblib")
    return tok, enc, clf

@st.cache_resource
def load_bert_ft():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_id = "Simingasa/fake-news-bert-finetuned"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tok, mdl



def embed_mean(enc_out, attn_mask):
    # enc_out: last_hidden_state [B,T,H]; attn_mask [B,T]
    mask = attn_mask.unsqueeze(-1)
    return (enc_out * mask).sum(1) / mask.sum(1).clamp(min=1)

def predict(model_name, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "TF-IDF + LogReg":
        pipe = load_tfidf()
        proba = pipe.predict_proba([text])[0]
        pred  = int(np.argmax(proba))
        return {"REAL": float(proba[0]), "FAKE": float(proba[1])}, pred
    elif model_name == "BERT (frozen) + LogReg":
        tok, enc, clf = load_bert_base()
        enc.to(device).eval()
        with torch.no_grad():
            batch = tok([text], truncation=True, padding=True, return_tensors="pt").to(device)
            out = enc(**batch).last_hidden_state
            pooled = embed_mean(out, batch["attention_mask"]).cpu().numpy()
            p = clf.predict_proba(pooled)[0]
            pred = int(np.argmax(p))
            return {"REAL": float(p[0]), "FAKE": float(p[1])}, pred
    else:
        tok, mdl = load_bert_ft()
        mdl.to(device).eval()
        with torch.no_grad():
            batch = tok([text], truncation=True, padding=True, return_tensors="pt").to(device)
            probs = mdl(**batch).logits.softmax(-1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
            return {"REAL": float(probs[0]), "FAKE": float(probs[1])}, pred

def attention_heatmap(text):
    if not (MODELS/"bert_finetuned").exists():
        st.info("L’heatmap funziona solo con il modello BERT fine-tuned.")
        return
    tok, mdl = load_bert_ft()
    mdl.eval()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True)
        out = mdl(**enc, output_attentions=True)
    attn = torch.stack(out.attentions).squeeze(1)      # [layers, heads, seq, seq]
    attn = attn.mean(dim=1).mean(dim=0).cpu().numpy()  # media su heads, poi layers -> [seq, seq]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    fig, ax = plt.subplots(figsize=(min(12, 0.5*len(tokens)+4), 6))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, ax=ax)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    st.pyplot(fig)

c1, c2 = st.columns([2,1])
with c1:
    if st.button("Classifica"):
        if text.strip():
            proba, pred = predict(model_name, text)
            st.metric("Predizione", "FAKE" if pred==1 else "REAL")
            st.write(proba)
        else:
            st.warning("Inserisci del testo.")
with c2:
    st.subheader("Grafici (se presenti)")
    for img in ["compare_roc.png", "tfidf_roc.png", "bert_base_roc.png"]:
        p = PLOTS/img
        if p.exists(): st.image(str(p), caption=img)

st.divider()
st.subheader("Attention heatmap (solo BERT fine-tuned)")
if st.button("Genera heatmap"):
    if text.strip():
        attention_heatmap(text)
    else:
        st.warning("Inserisci del testo.")
