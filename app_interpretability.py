# app_interpretability_paper_grade.py
# Drop-in replacement for your app_interpretability.py, refactored for
# paper-grade interpretability demos on fake-news detection.
#
# Key improvements
# - Reproducibility (seed)
# - Device safety & caching
# - Wordpiece -> word aggregation (handles BERT/WordPiece & (XLM-)RoBERTa/SentencePiece)
# - Special-token filtering throughout ([CLS]/[SEP]/[PAD]/etc.)
# - Signed attributions (direction + magnitude) and centered visualization
# - Attention Rollout (Abnar & Zuidema, 2020) implemented carefully
# - Gradient×Input and Integrated Gradients (with PAD baseline) on logits
# - Occlusion using Δ logit for the predicted class (more faithful than Δ prob)
# - Completeness check for IG (displays residual)
# - Safer SHAP fallback
# - Clean Streamlit UI with controls

from __future__ import annotations
import os, json, random, re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Optional deps
try:
    from captum.attr import IntegratedGradients
    HAVE_CAPTUM = True
except Exception:
    HAVE_CAPTUM = False

try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

# ---------------------------
# Basics & Reproducibility
# ---------------------------
BASE = Path(__file__).parent.resolve()
MODELS = BASE / "models"
PLOTS = BASE / "plots"

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="📰 Fake News Detector — Interpretability (Paper Grade)", layout="wide")
st.title("📰 Fake News Detector — Interpretability (Paper Grade)")

available = {
    "TF-IDF + LogReg": (MODELS/"tfidf_logreg.joblib").exists(),
    "BERT (frozen) + LogReg": (MODELS/"bert_base_logreg.joblib").exists() and (MODELS/"bert_base_config.json").exists(),
    "BERT (fine-tuned)": (MODELS/"bert_finetuned").exists()
}
choices = [k for k,v in available.items() if v]
if not choices:
    st.warning("Nessun modello trovato in /models. Addestra prima i modelli.")
    st.stop()

with st.sidebar:
    model_name = st.selectbox("Seleziona modello", choices)
    center_zero = st.checkbox("Colori divergenti (mostra segno ±)", value=True)
    drop_special = st.checkbox("Escludi token speciali ([CLS], [SEP], [PAD], …)", value=True)
    merge_wordpieces_flag = st.checkbox("Aggrega subword → parole", value=True)

text = st.text_area("Incolla un articolo:", height=220, placeholder="Titolo e/o testo dell'articolo…")

# ---------------------------
# Loading & Device
# ---------------------------
@st.cache_resource(show_spinner=False)
def device_choice() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = device_choice()

def _to(x):
    return x.to(DEVICE) if isinstance(x, torch.nn.Module) else {k: v.to(DEVICE) for k,v in x.items()}

@st.cache_resource(show_spinner=False)
def load_tfidf():
    return joblib.load(MODELS/"tfidf_logreg.joblib")

@st.cache_resource(show_spinner=False)
def load_bert_base():
    cfg = json.loads((MODELS/"bert_base_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["hf_model"], use_fast=True)
    enc = AutoModel.from_pretrained(cfg["hf_model"]).eval()
    clf = joblib.load(MODELS/"bert_base_logreg.joblib")
    return tok, enc, clf

@st.cache_resource(show_spinner=False)
def load_bert_ft():
    model_dir = MODELS/"bert_finetuned"
    if model_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    else:
        # Fallback ONLY for demo; replace with your own FT model for paper runs.
        model_id = "Simingasa/fake-news-bert-finetuned"
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    return tok, mdl

# ---------------------------
# Token utilities
# ---------------------------
SPECIAL_RE = re.compile(r"^(<s>|</s>|<pad>|<unk>|\[CLS\]|\[SEP\]|\[PAD\]|\[UNK\]|</w>)$", re.IGNORECASE)


def is_special(token: str) -> bool:
    return bool(SPECIAL_RE.match(token))


def merge_wordpieces(tokens: List[str], scores: np.ndarray, tok) -> Tuple[List[str], np.ndarray]:
    """Merge WordPiece (##) and SentencePiece (▁) tokens into words and aggregate scores.
    Heuristics cover BERT, RoBERTa, XLM-R, etc.
    Aggregation: sum of token scores per word (you may switch to mean via flag if desired)."""
    if not len(tokens):
        return tokens, scores

    # Identify scheme
    uses_hash = any(t.startswith("##") for t in tokens)
    uses_underline = any("▁" in t for t in tokens)  # SentencePiece (XLM-R/roberta)
    uses_G = any(t.startswith("Ġ") for t in tokens)  # RoBERTa variant

    merged_tokens: List[str] = []
    merged_scores: List[float] = []

    buf = []
    buf_score = 0.0

    def flush():
        nonlocal buf, buf_score
        if buf:
            merged_tokens.append("".join(buf))
            merged_scores.append(buf_score)
            buf, buf_score = [], 0.0

    for t, s in zip(tokens, scores):
        if drop_special and is_special(t):
            flush()
            continue
        if uses_hash and t.startswith("##"):
            buf.append(t[2:])
            buf_score += float(s)
        elif uses_underline:
            # SentencePiece: '▁' marks word start
            if t.startswith("▁"):
                flush()
                buf = [t[1:]]
                buf_score = float(s)
            else:
                buf.append(t)
                buf_score += float(s)
        elif uses_G:
            # RoBERTa: 'Ġ' marks space/new word
            if t.startswith("Ġ"):
                flush()
                buf = [t[1:]]
                buf_score = float(s)
            else:
                buf.append(t)
                buf_score += float(s)
        else:
            # Plain tokenization: split on spaces
            flush()
            buf = [t]
            buf_score = float(s)
    flush()
    return merged_tokens, np.array(merged_scores, dtype=float)


def apply_filters(tokens: List[str], scores: np.ndarray, tok) -> Tuple[List[str], np.ndarray]:
    toks, sc = tokens, np.array(scores, dtype=float)
    if drop_special:
        keep = [not is_special(t) for t in toks]
        toks = [t for t,k in zip(toks, keep) if k]
        sc = sc[keep]
    if merge_wordpieces_flag:
        toks, sc = merge_wordpieces(toks, sc, tok)
    return toks, sc

# ---------------------------
# Prediction helper (consistent logits/probs)
# ---------------------------

def embed_mean(enc_out, attn_mask):
    mask = attn_mask.unsqueeze(-1)
    return (enc_out * mask).sum(1) / mask.sum(1).clamp(min=1)


def predict(model_name: str, text: str):
    if model_name == "TF-IDF + LogReg":
        pipe = load_tfidf()
        proba = pipe.predict_proba([text])[0]
        pred = int(np.argmax(proba))
        return {"REAL": float(proba[0]), "FAKE": float(proba[1])}, pred, None

    elif model_name == "BERT (frozen) + LogReg":
        tok, enc, clf = load_bert_base()
        with torch.no_grad():
            batch = tok([text], truncation=True, padding=True, return_tensors="pt")
            batch = _to(batch)
            enc = _to(enc)
            out = enc(**batch).last_hidden_state
            pooled = embed_mean(out, batch["attention_mask"]).cpu().numpy()
            p = clf.predict_proba(pooled)[0]
            pred = int(np.argmax(p))
            return {"REAL": float(p[0]), "FAKE": float(p[1])}, pred, tok

    else:
        tok, mdl = load_bert_ft()
        mdl = _to(mdl)
        with torch.no_grad():
            batch = tok([text], truncation=True, padding=True, return_tensors="pt")
            batch = _to(batch)
            logits = mdl(**batch).logits[0]
            probs = logits.softmax(-1).cpu().numpy()
            pred = int(np.argmax(probs))
            return {"REAL": float(probs[0]), "FAKE": float(probs[1])}, pred, tok

# ---------------------------
# Visualization helpers
# ---------------------------

def token_bar(tokens: List[str], scores: np.ndarray, title: str, center_zero: bool = True, sort_by_abs: bool = False):
    vals = np.array(scores, dtype=float)
    toks = list(tokens)
    if sort_by_abs:
        order = np.argsort(-np.abs(vals))
        vals = vals[order]
        toks = [toks[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.28*len(toks))))

    if center_zero:
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in vals]
    else:
        # Normalize 0..1 for non-diverging
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (vals - vmin) / (max(vmax - vmin, 1e-8))
        from matplotlib.cm import YlOrRd
        colors = [YlOrRd(x) for x in norm]

    y = np.arange(len(toks))
    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(toks, fontsize=9)
    ax.set_xlabel('Score')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    if center_zero:
        ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------
# 1) Attention heatmap (token-row view) — INFO-ONLY, not an explanation
# ---------------------------

def attention_heatmap(text: str):
    tok, mdl = load_bert_ft()
    mdl = _to(mdl).eval()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True)
        enc = _to(enc)
        out = mdl(**enc, output_attentions=True)

    attn = torch.stack(out.attentions)  # [L,B,H,S,S]
    attn_avg = attn.mean(dim=2).mean(dim=0)[0]  # [S,S]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu())

    st.write("### Seleziona un token per vedere la sua attention (media su layer/head)")
    token_idx = st.selectbox(
        "Token:", list(range(len(tokens))), format_func=lambda i: f"{i}: {tokens[i]}"
    )
    row = attn_avg[token_idx, :].detach().cpu().numpy()

    # Apply filters/merge to the column tokens and map row accordingly
    base_tokens = tokens
    base_scores = row

    # Build mask of kept positions after filtering specials; merging handled after
    if drop_special:
        keep_mask = np.array([not is_special(t) for t in base_tokens], dtype=bool)
    else:
        keep_mask = np.ones(len(base_tokens), dtype=bool)

    kept_tokens = [t for t,k in zip(base_tokens, keep_mask) if k]
    kept_scores = base_scores[keep_mask]

    if merge_wordpieces_flag:
        kept_tokens, kept_scores = merge_wordpieces(kept_tokens, kept_scores, tok)

    token_bar(kept_tokens, kept_scores, f"Attention da '{tokens[token_idx]}'", center_zero=False)

# ---------------------------
# 2) Attention Rollout (Abnar & Zuidema, 2020) — CLS→token connectivity
# ---------------------------

def attention_rollout_importance(text: str):
    tok, mdl = load_bert_ft()
    mdl = _to(mdl).eval()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True)
        enc = _to(enc)
        out = mdl(**enc, output_attentions=True)

    attn = torch.stack(out.attentions)  # [L,B,H,S,S]
    attn = attn.mean(dim=2)[..., 0]     # avg heads; take CLS row → [L,B,S]
    attn = attn[:, 0, :]                # [L,S]

    S = attn.shape[-1]
    I = torch.eye(S, device=attn.device)
    rollout = I
    for A in attn:
        A = A + I  # add skip connection
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A.T @ rollout

    cls_to_tok = rollout[0].detach().cpu().numpy()
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu())
    tokens, cls_to_tok = apply_filters(tokens, cls_to_tok, tok)
    token_bar(tokens, cls_to_tok, "Attention Rollout: CLS→token", center_zero=False, sort_by_abs=True)

# ---------------------------
# 3) Gradient×Input (signed) on logit of predicted class
# ---------------------------

def gradient_x_input(text: str):
    if model_name != "BERT (fine-tuned)":
        st.info("Richiede un BERT fine-tuned per i gradienti.")
        return
    tok, mdl = load_bert_ft()
    mdl = _to(mdl).eval()

    enc = tok(text, return_tensors="pt", truncation=True, padding=True)
    enc = _to(enc)
    emb_layer = mdl.get_input_embeddings()

    # Build embeddings with grad
    embeds = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
    logits = mdl(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits
    pred = logits.argmax(-1)
    target_logit = logits[0, pred]
    mdl.zero_grad()
    target_logit.backward()

    grads = embeds.grad.detach()  # [1,S,H]
    atts = (grads * embeds).sum(dim=-1)[0]  # signed grad×input per token
    scores = atts.cpu().numpy()

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu())
    tokens, scores = apply_filters(tokens, scores, tok)
    token_bar(tokens, scores, "Gradient×Input (logit)", center_zero=True, sort_by_abs=True)

# ---------------------------
# 4) Integrated Gradients (PAD baseline) + completeness residual
# ---------------------------

def integrated_gradients(text: str, steps: int = 32):
    if model_name != "BERT (fine-tuned)":
        st.info("Richiede un BERT fine-tuned.")
        return
    if not HAVE_CAPTUM:
        st.info("Captum non installato: `pip install captum`." )
        return

    tok, mdl = load_bert_ft()
    mdl = _to(mdl).eval()

    enc = tok(text, return_tensors="pt", truncation=True, padding=True)
    enc = _to(enc)

    emb_layer = mdl.get_input_embeddings()
    inputs = emb_layer(enc["input_ids"])  # [1,S,H]

    # Baseline: PAD embedding (or zeros if not available)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    baseline_embed = emb_layer(torch.tensor([[pad_id]*inputs.shape[1]], device=inputs.device))

    def fwd(e):
        return mdl(inputs_embeds=e, attention_mask=enc["attention_mask"]).logits

    with torch.no_grad():
        logits = fwd(inputs)
        pred = logits.argmax(-1)
        base_logits = fwd(baseline_embed)

    ig = IntegratedGradients(lambda e: fwd(e)[:, pred])
    attributions, delta = ig.attribute(inputs, baselines=baseline_embed, return_convergence_delta=True, n_steps=steps)

    # Signed aggregation over embedding dim
    tok_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()

    # Completeness (residual)
    with torch.no_grad():
        residual = (logits[0, pred] - base_logits[0, pred] - attributions.sum()).item()

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu())
    tokens, tok_scores = apply_filters(tokens, tok_scores, tok)
    token_bar(tokens, tok_scores, f"Integrated Gradients (steps={steps})", center_zero=True, sort_by_abs=True)
    st.caption(f"IG completeness residual (should be ~0): {residual:.4f}")

# ---------------------------
# 5) Occlusion / masking — Δ logit for predicted class
# ---------------------------

def occlusion_importance(text: str, max_tokens: int = 160):
    proba, pred, tok_or_none = predict(model_name, text)

    if model_name == "TF-IDF + LogReg":
        pipe = load_tfidf()
        vec = pipe.named_steps.get('tfidf') or pipe.named_steps.get('vectorizer')
        clf = pipe.named_steps.get('logreg') or pipe.named_steps.get('classifier')
        tokens = text.split()[:max_tokens]
        base_X = vec.transform([text])
        base_logit = float(base_X @ clf.coef_[pred].reshape(-1,1) + clf.intercept_[pred])
        scores = []
        for i in range(len(tokens)):
            txt = " ".join(tokens[:i] + tokens[i+1:])
            X = vec.transform([txt])
            logit = float(X @ clf.coef_[pred].reshape(-1,1) + clf.intercept_[pred])
            scores.append(base_logit - logit)
        token_bar(tokens, np.array(scores), "Occlusion Δ logit (remove token)", center_zero=True, sort_by_abs=True)
        return

    # BERT models
    if tok_or_none is None:
        tok, mdl = load_bert_ft()
    else:
        tok = tok_or_none
        _, mdl = load_bert_ft()
    mdl = _to(mdl).eval()

    with torch.no_grad():
        base = tok([text], truncation=True, padding=True, return_tensors="pt")
        base = _to(base)
        base_logits = mdl(**base).logits[0]
        base_pred = int(base_logits.argmax().item())
        base_logit = float(base_logits[base_pred].item())

    input_ids = base["input_ids"][0].tolist()
    tokens = tok.convert_ids_to_tokens(input_ids)

    # Apply filters but keep an index mapping to input_ids for masking
    keep = [True]*len(tokens)
    if drop_special:
        keep = [not is_special(t) for t in tokens]
    kept_indices = [i for i,k in enumerate(keep) if k][:max_tokens]

    scores = []
    for i in kept_indices:
        ids = input_ids.copy()
        if tok.mask_token_id is not None:
            ids[i] = tok.mask_token_id
        else:
            # delete the token
            del ids[i]
        b = tok.prepare_for_model(ids, return_tensors="pt")
        b = _to(b)
        with torch.no_grad():
            logit = float(mdl(**b).logits[0, base_pred].item())
        scores.append(base_logit - logit)

    kept_tokens = [tokens[i] for i in kept_indices]
    if merge_wordpieces_flag:
        kept_tokens, scores = merge_wordpieces(kept_tokens, np.array(scores), tok)
    token_bar(kept_tokens, np.array(scores), "Occlusion Δ logit (mask/remove)", center_zero=True, sort_by_abs=True)

# ---------------------------
# 6) TF-IDF + LogReg: exact local feature contributions
# ---------------------------

def tfidf_local_explanations(text: str):
    pipe = load_tfidf()
    vec = pipe.named_steps.get('tfidf') or pipe.named_steps.get('vectorizer')
    clf = pipe.named_steps.get('logreg') or pipe.named_steps.get('classifier')
    if vec is None or clf is None:
        st.warning("Pipeline non in formato atteso (tfidf, logreg).")
        return
    X = vec.transform([text])  # sparse
    pred = int(np.argmax(clf.predict_proba(X)[0]))
    # contributions = value * coef[class]
    X_csr = X.tocoo()
    inv_vocab = {i:t for t,i in vec.vocabulary_.items()}
    contribs = {}
    for i, j, v in zip(X_csr.row, X_csr.col, X_csr.data):
        tok = inv_vocab.get(j, f"f{j}")
        contribs[tok] = float(v * clf.coef_[pred, j])
    if not contribs:
        st.info("Nessuna feature riconosciuta nel testo.")
        return
    items = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    top_k = 20
    pos = items[:top_k]
    neg = items[-top_k:]

    st.write("**Top feature che spingono la classe predetta**")
    token_bar([k for k,_ in pos], np.array([v for _,v in pos]), "TF-IDF contributi (+)", center_zero=True)
    st.write("**Top feature che la ostacolano**")
    token_bar([k for k,_ in neg], np.array([v for _,v in neg]), "TF-IDF contributi (−)", center_zero=True)

# ---------------------------
# 7) SHAP (optional, slow) — kernel on text masker
# ---------------------------

def shap_explain(text: str, nsamples: int = 200):
    if not HAVE_SHAP:
        st.info("SHAP non installato: `pip install shap`.")
        return

    if model_name == "TF-IDF + LogReg":
        pipe = load_tfidf()
        vec = pipe.named_steps.get('tfidf') or pipe.named_steps.get('vectorizer')
        clf = pipe.named_steps.get('logreg') or pipe.named_steps.get('classifier')
        explainer = shap.LinearExplainer(clf, vec.transform([""]))
        X = vec.transform([text])
        sv = explainer.shap_values(X)
        # Approx token-level via analyzer vocab mapping
        tokens = vec.build_analyzer()(text)
        token_scores = []
        pred = int(np.argmax(clf.predict_proba(X)[0]))
        for t in tokens:
            j = vec.vocabulary_.get(t)
            if j is not None:
                token_scores.append((t, float(clf.coef_[pred, j])))
        if token_scores:
            toks, scores = zip(*token_scores)
            token_bar(list(toks), np.array(scores), "TF-IDF proxy SHAP/coef", center_zero=True)
        else:
            st.info("Nessun token SHAP mappato.")
        return

    # For BERT, use text masker and model wrapper returning probs
    def f(batch_texts):
        outs = []
        for t in batch_texts:
            p, _, _ = predict("BERT (fine-tuned)", t)
            outs.append([p["REAL"], p["FAKE"]])
        return np.array(outs)

    masker = shap.maskers.Text(" ")
    explainer = shap.Explainer(f, masker)
    sv = explainer([text], max_evals=nsamples)
    try:
        html = shap.plots.text(sv, display=False)
        st.components.v1.html(html.data, height=300, scrolling=True)
    except Exception:
        st.info("Rendering HTML SHAP non supportato; mostro importanze aggregate.")
        pred = int(np.argmax(f([text])[0]))
        vals = sv.values[0, :, pred]
        toks = sv.data[0]
        token_bar(list(toks), np.array(vals), "SHAP (kernel) token scores", center_zero=True)

# -----------------
# UI — prediction & metrics
# -----------------

c1, c2 = st.columns([2,1])
with c1:
    if st.button("Classifica"):
        if text.strip():
            proba, pred, _ = predict(model_name, text)
            st.metric("Predizione", "FAKE" if pred==1 else "REAL")
            st.json(proba)
        else:
            st.warning("Inserisci del testo.")
with c2:
    st.subheader("Grafici ROC (se presenti)")
    for img in ["compare_roc.png", "tfidf_roc.png", "bert_base_roc.png"]:
        p = PLOTS/img
        if p.exists():
            st.image(str(p), caption=img)

st.divider()
st.subheader("Interpretabilità — scegli uno strumento")

att_tab, roll_tab, gxi_tab, ig_tab, occ_tab, tfidf_tab, shap_tab = st.tabs([
    "Attention (media)",
    "Attention Rollout",
    "Gradient×Input",
    "Integrated Gradients",
    "Occlusion",
    "TF-IDF contributi",
    "SHAP (opzionale)"
])

with att_tab:
    st.write("**Nota**: le matrici di attenzione non sono spiegazioni fedeli; usale solo come indizio.")
    if st.button("Mostra attention per token"):
        if "BERT" not in model_name:
            st.info("Usa un modello BERT per la heatmap.")
        elif text.strip():
            attention_heatmap(text)
        else:
            st.warning("Inserisci del testo.")

with roll_tab:
    if st.button("Calcola Rollout CLS→token"):
        if "BERT" not in model_name:
            st.info("Usa un modello BERT per il rollout.")
        elif text.strip():
            attention_rollout_importance(text)
        else:
            st.warning("Inserisci del testo.")

with gxi_tab:
    if st.button("Gradient×Input"):
        if text.strip():
            gradient_x_input(text)
        else:
            st.warning("Inserisci del testo.")

with ig_tab:
    steps = st.slider("Passi IG", min_value=8, max_value=128, value=32, step=4)
    if st.button("Integrated Gradients"):
        if text.strip():
            integrated_gradients(text, steps=steps)
        else:
            st.warning("Inserisci del testo.")

with occ_tab:
    max_tokens = st.slider("Max token da analizzare", 20, 240, 160, 10)
    if st.button("Occlusion / masking"):
        if text.strip():
            occlusion_importance(text, max_tokens=max_tokens)
        else:
            st.warning("Inserisci del testo.")

with tfidf_tab:
    if st.button("Contributi TF-IDF"):
        if model_name == "TF-IDF + LogReg" and text.strip():
            tfidf_local_explanations(text)
        elif model_name != "TF-IDF + LogReg":
            st.info("Seleziona il modello TF-IDF + LogReg.")
        else:
            st.warning("Inserisci del testo.")

with shap_tab:
    ns = st.slider("Campioni (approx)", 50, 1000, 200, 50)
    if st.button("Calcola SHAP"):
        if text.strip():
            shap_explain(text, nsamples=ns)
        else:
            st.warning("Inserisci del testo.")

st.caption("Suggerimenti: 1) Usa testi di almeno 2–3 frasi; 2) Per paper, riporta anche deletion/insertion curves come test di fedeltà.")
