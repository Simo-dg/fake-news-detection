import os
import re
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from duckduckgo_search import DDGS
from matplotlib.cm import Blues
from urllib.parse import urlparse
from bertopic import BERTopic
from src import config

# --- CONFIGURATION ---
st.set_page_config(page_title="🛡️ TruthLens Command Center", layout="wide", initial_sidebar_state="collapsed")

MODELS = config.MODELS_DIR

# Model Paths
HF_MODEL_ID = config.HF_MODEL_ID
HF_MODEL_ID_V2 = config.HF_MODEL_ID_V2
LOCAL_MODEL_PATH = MODELS / "bert_final"

DEVICE = config.DEVICE
MAX_LENGTH = config.MAX_LENGTH
CHUNK_OVERLAP = config.CHUNK_OVERLAP

# --- IMPROVED CSS STYLING (DARK MODE COMPATIBLE) ---
st.markdown("""
<style>
    /* 1. Remove hardcoded backgrounds on main to allow Dark Mode to work naturally */
    /* 2. Fix Metric styling to adapt to theme or look good on both */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1); /* Semi-transparent grey */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* 3. Badges: Keep these hardcoded as they have specific background colors */
    .verdict-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge-sup { background-color: #e6f4ea; color: #1e8e3e; border: 1px solid #1e8e3e; }
    .badge-ref { background-color: #fce8e6; color: #d93025; border: 1px solid #d93025; }
    .badge-neu { background-color: #f1f3f4; color: #5f6368; border: 1px solid #5f6368; }

    /* 4. Source Card: Force BLACK text on WHITE background regardless of Theme */
    .source-card { 
        background-color: #ffffff; 
        color: #31333F; /* Force dark text */
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        border: 1px solid #ddd; 
    }
    
    /* Ensure links inside source cards are visible on white background */
    .source-card a {
        color: #1565c0 !important;
    }
    
    .source-card .snippet {
        color: #555555 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1) CLEANING FUNCTION (V12 STRICT) ---
def clean_text_bert(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    
    patterns = [
        r'advertisement', r'reading main', r'main story', r'continue reading',
        r'president elect', r'source text', r'rights reserved', 
        r'copyright', r'misstated', r'company coverage', r'newsletter',
        r'fact box', r'story fact', r'fact check', r'add cents', 
        r'add your two cents', r'readers think', r'view gallery', 
        r'featured image', r'read more', r'click here', r'sign up', 
        r'proactiveinvestors', r'visit our', r'check out'
    ]
    combined_pattern = re.compile('|'.join(patterns))
    text = combined_pattern.sub(' ', text)

    meta_words = [
        r'\bphoto\b', r'\bimage\b', r'\bcredit\b', r'\beditor\b', 
        r'\bediting\b', r'\bwriter\b', r'\bphotograph\b', 
        r'\bcaption\b', r'\breporting\b'
    ]
    for p in meta_words: text = re.sub(p, ' ', text)

    agencies = [
        r'\b(reuters|ap|afp|upi|bloomberg|cnbc|cnn|bbc|nyt|new york times|washington post)\b',
        r'\b(nytimes|breitbart|christian post|consortiumnews|daily caller)\b',
        r'\b(calif|gmt|est|pst)\b'
    ]
    for pattern in agencies: text = re.sub(pattern, ' ', text)

    months = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b'
    text = re.sub(months, ' ', text)

    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2) LOADERS ---

@st.cache_resource
def load_bert_models():
    models = {}
    try:
        tok_old = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        mdl_old = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID).to(DEVICE)
        mdl_old.eval()
        models['legacy'] = (tok_old, mdl_old)
    except Exception: pass

    try:
        tok_new = AutoTokenizer.from_pretrained(HF_MODEL_ID_V2)
        mdl_new = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID_V2).to(DEVICE)
        mdl_new.eval()
        models['new'] = (tok_new, mdl_new)
    except Exception: pass
    return models

@st.cache_resource
def load_tfidf_model():
    path = MODELS / "tfidf_logreg_robust.joblib"
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_nli_model():
    return CrossEncoder('cross-encoder/nli-deberta-v3-small')

@st.cache_resource
def load_bertopic_model():
    path = MODELS / "bertopic_model"
    if path.exists():
        return BERTopic.load(path)
    return None

# --- 3) ANALYTICS ---

def analyze_bert(text, tok, mdl):
    encoded = tok(text, truncation=True, max_length=MAX_LENGTH, stride=CHUNK_OVERLAP, 
                 return_overflowing_tokens=True, padding=False)
    chunk_logits, attentions, input_ids_list = [], [], []

    with torch.no_grad():
        for i in range(len(encoded["input_ids"])):
            inp_ids = torch.tensor([encoded["input_ids"][i]]).to(DEVICE)
            mask = torch.tensor([encoded["attention_mask"][i]]).to(DEVICE)
            try:
                out = mdl(input_ids=inp_ids, attention_mask=mask, output_attentions=True)
                chunk_logits.append(out.logits[0].cpu().numpy())
                attn = torch.stack(out.attentions)[-1].mean(dim=1)[0]
                attentions.append(attn.sum(dim=0).cpu().numpy())
            except:
                out = mdl(input_ids=inp_ids, attention_mask=mask)
                chunk_logits.append(out.logits[0].cpu().numpy())
                attentions.append(np.zeros_like(inp_ids[0].cpu().numpy(), dtype=float))
            input_ids_list.append(encoded["input_ids"][i])

    avg_logits = np.mean(chunk_logits, axis=0)
    exps = np.exp(avg_logits - np.max(avg_logits))
    probs = exps / np.sum(exps)
    pred = int(np.argmax(avg_logits))
    tokens = tok.convert_ids_to_tokens(input_ids_list[0])

    return {
        "pred": pred,
        "prob_fake": float(probs[1]),
        "tokens": tokens,
        "attention": attentions[0] if attentions else np.zeros(len(tokens))
    }

def analyze_tfidf(text, pipeline):
    if pipeline is None: return None
    try:
        probs = pipeline.predict_proba([text])[0]
        pred = int(np.argmax(probs))
        
        vectorizer = pipeline.named_steps.get('tfidf', pipeline.named_steps.get('vectorizer'))
        clf = pipeline.named_steps.get('clf', pipeline.named_steps.get('logreg'))
        
        if not vectorizer or not clf:
            return {"pred": pred, "prob_fake": float(probs[1]), "features": []}

        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]
        response = vectorizer.transform([text]).tocoo()
        feats = [(feature_names[col], val * coefs[col]) for col, val in zip(response.col, response.data)]
        feats.sort(key=lambda x: abs(x[1]), reverse=True)
        return {"pred": pred, "prob_fake": float(probs[1]), "features": feats[:20]}
    except: return None

# --- 4) SEARCH & FACT CHECK ---

def perform_search(q, region="us-en", timelimit=None):
    results = []
    try:
        with DDGS() as ddgs:
            kwargs = dict(region=region, safesearch='moderate', max_results=5)
            if timelimit: kwargs["timelimit"] = timelimit
            ddg_results = list(ddgs.news(q, **kwargs))
            for r in ddg_results:
                body = r.get('body', r.get('snippet', ''))
                if body:
                    results.append({
                        "text": body, 
                        "url": r.get('url', r.get('link', '')), 
                        "title": r.get('title', 'Untitled'), 
                        "source": r.get('source', 'Web')
                    })
    except: pass
    return results

def verify_facts(text, nli_model, region="us-en", timelimit=None):
    claim = text[:250]
    raw_results = perform_search(claim, region=region, timelimit=timelimit)
    
    if not raw_results:
        return {"status": "NO_DATA", "verdict": "UNVERIFIED", "confidence": 0.0, "evidence": []}

    processed_evidence = []
    pairs = [(claim, r['text']) for r in raw_results]
    logits = nli_model.predict(pairs)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    total_entail = 0.0
    total_contra = 0.0

    for i, r in enumerate(raw_results):
        p_contra = probs[i][0]
        p_entail = probs[i][1]
        p_neut = probs[i][2]
        
        if p_entail > 0.5: 
            label = "SUPPORT"
            score = p_entail
            badge_class = "badge-sup"
        elif p_contra > 0.5: 
            label = "REFUTE"
            score = p_contra
            badge_class = "badge-ref"
        else: 
            label = "NEUTRAL"
            score = p_neut
            badge_class = "badge-neu"
            
        processed_evidence.append({
            "title": r['title'],
            "source": r['source'],
            "url": r['url'],
            "snippet": r['text'],
            "label": label,
            "score": score,
            "badge_class": badge_class
        })
        total_entail += p_entail
        total_contra += p_contra

    avg_entail = total_entail / len(raw_results)
    avg_contra = total_contra / len(raw_results)
    
    if avg_entail > 0.4 and avg_entail > avg_contra:
        global_verdict = "VERIFIED"
        global_conf = avg_entail
    elif avg_contra > 0.4 and avg_contra > avg_entail:
        global_verdict = "DEBUNKED"
        global_conf = avg_contra
    else:
        global_verdict = "INCONCLUSIVE"
        global_conf = max(avg_entail, avg_contra)

    return {
        "status": "SUCCESS",
        "verdict": global_verdict,
        "confidence": global_conf,
        "evidence": processed_evidence 
    }

# --- 5) VISUALIZATIONS (Updated for Visibility) ---

def plot_attention_bar(tokens, scores):
    clean = [(t, s) for t, s in zip(tokens, scores) if t not in ['[CLS]', '[SEP]', '[PAD]']]
    if not clean: return plt.figure()
    c_tokens, c_scores = zip(*clean)
    
    # Facecolor white ensures labels are black and visible on dark backgrounds
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='white') 
    ax.bar(range(len(c_tokens[:20])), c_scores[:20], color=Blues(c_scores[:20]))
    ax.set_xticks(range(len(c_tokens[:20])))
    ax.set_xticklabels(c_tokens[:20], rotation=45, ha='right', color='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    return fig

def plot_tfidf_importance(features):
    if not features: return plt.figure()
    words = [f[0] for f in features[:15]][::-1]
    scores = [f[1] for f in features[:15]][::-1]
    
    # Facecolor white ensures labels are black and visible on dark backgrounds
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    colors = ['#d32f2f' if s > 0 else '#388e3c' for s in scores]
    ax.barh(words, scores, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    return fig

# --- 6) MAIN DASHBOARD ---

st.title("🛡️ TruthLens: Forensic News Dashboard")

with st.sidebar:
    st.header("Configuration")
    # Region selection removed
    recency_sel = st.selectbox("Recency", ["Any time", "Past day", "Past week", "Past month"], index=0)
    recency_map = {"Any time": None, "Past day": "d", "Past week": "w", "Past month": "m"}

text = st.text_area("Input Article Text", height=160, placeholder="Paste text here...")

if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please enter text.")
    else:
        bert_models = load_bert_models()
        tfidf_pipe = load_tfidf_model()
        nli_mdl = load_nli_model()
        bertopic_model = load_bertopic_model()
        
        with st.spinner("🕵️‍♀️ Running Dual-Core Analysis..."):
            # 1. Legacy BERT
            bert_res_old = None
            if 'legacy' in bert_models:
                bert_res_old = analyze_bert(text, bert_models['legacy'][0], bert_models['legacy'][1])
            
            # 2. New V12 BERT (Cleaned)
            bert_res_new = None
            if 'new' in bert_models:
                cleaned_text = clean_text_bert(text)
                bert_res_new = analyze_bert(cleaned_text, bert_models['new'][0], bert_models['new'][1])

            # 3. TF-IDF
            tfidf_res = analyze_tfidf(text, tfidf_pipe)
            
            # 4. Fact Check
            fact_res = verify_facts(text, nli_mdl, region="us-en", timelimit=recency_map[recency_sel])

            # 5. BERTopic
            topic_viz = None
            topic_info = None
            if bertopic_model:
                topics, probs = bertopic_model.transform([text])
                topic_id = topics[0]
                topic_viz = bertopic_model.visualize_barchart(top_n_topics=8, topics=[topic_id])
                topic_info = bertopic_model.get_topic_info(topic_id)

        st.divider()

        # --- METRICS GRID ---
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("### ✅ V1 BERT")
            if bert_res_old:
                lbl = "FAKE" if bert_res_old['pred'] == 1 else "REAL"
                st.metric("Model Verdict", lbl, f"{bert_res_old['prob_fake']:.1%} fake-prob", 
                          delta_color="inverse" if lbl == "FAKE" else "normal")

        with c2:
            st.markdown("### ✈️ V2 BERT")
            if bert_res_new:
                lbl = "FAKE" if bert_res_new['pred'] == 1 else "REAL"
                st.metric("Model Verdict", lbl, f"{bert_res_new['prob_fake']:.1%} fake-prob", 
                          delta_color="inverse" if lbl == "FAKE" else "normal")

        with c3:
            st.markdown("### 🧮 TF-IDF")
            if tfidf_res:
                lbl = "FAKE" if tfidf_res['pred'] == 1 else "REAL"
                st.metric("Heuristic", lbl, f"{tfidf_res['prob_fake']:.1%} fake-prob", 
                          delta_color="inverse" if lbl == "FAKE" else "normal")

        with c4:
            st.markdown("### 🌍 Fact Check")
            v = fact_res['verdict']
            color = "normal" if v == "VERIFIED" else "inverse"
            st.metric("Cross-Ref", v, f"{fact_res['confidence']:.1%}", delta_color=color)

        # --- TABS ---
        t1, t2, t3, t4 = st.tabs(["🧠 Attention", "🧮 Features", "🌍 Evidence & Sources", "🗂️ Semantic Topic"])
        
        with t1:
            c_old, c_new = st.columns(2)
            with c_old:
                st.markdown("**Bert V1 Attention**")
                if bert_res_old: st.pyplot(plot_attention_bar(bert_res_old['tokens'], bert_res_old['attention']))
            with c_new:
                st.markdown("**Bert V2 Attention**")
                if bert_res_new: st.pyplot(plot_attention_bar(bert_res_new['tokens'], bert_res_new['attention']))

        with t2:
            if tfidf_res:
                st.markdown("**TF-IDF Top Features**")
                st.pyplot(plot_tfidf_importance(tfidf_res['features']))

        with t3:
            st.markdown("#### Web Verification Results")
            if fact_res['status'] == "NO_DATA":
                st.warning("No search results found.")
            else:
                for item in fact_res['evidence']:
                    badge = f'<span class="verdict-badge {item["badge_class"]}">{item["label"]} ({item["score"]:.0%})</span>'
                    
                    # Updated HTML with forced colors to handle white background in dark mode
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>{item['source']}</b>
                            {badge}
                        </div>
                        <div style="font-size:1.1em; margin-top:5px;">
                            <a href="{item['url']}" target="_blank" style="text-decoration:none; color: #1565c0;">{item['title']}</a>
                        </div>
                        <div class="snippet" style="font-size:0.9em; color: #555555; margin-top:5px;">
                            {item['snippet'][:250]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with t4:
            if bertopic_model:
                st.subheader(f"Topic Identification: {topic_info['Name'].values[0]}")
                st.markdown(f"**Topic ID:** {topic_info['Topic'].values[0]} | **Count:** {topic_info['Count'].values[0]}")
                
                if topic_viz:
                    st.plotly_chart(topic_viz, use_container_width=True)
                
                with st.expander("View Raw Topic Keywords"):
                    st.dataframe(topic_info, use_container_width=True)
            else:
                st.warning("BERTopic model not loaded.")