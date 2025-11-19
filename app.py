# app.py
import os
import re
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from duckduckgo_search import DDGS
from matplotlib.cm import Blues
from urllib.parse import urlparse

# --- CONFIGURATION ---
st.set_page_config(page_title="🛡️ TruthLens Command Center", layout="wide", initial_sidebar_state="collapsed")

BASE = Path(__file__).parent.resolve()
MODELS = BASE / "models"
HF_MODEL_ID = "Simingasa/fake-news-bert-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
CHUNK_OVERLAP = 128

# --- CSS STYLING ---
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    h1 { color: #1E1E1E; font-family: 'Helvetica Neue', sans-serif; }
    .source-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; color: white; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .small { font-size: 0.9em; color: #666; }
</style>
""", unsafe_allow_html=True)

# --- 1) LOADERS (CACHED) ---

@st.cache_resource
def load_bert_model():
    try:
        with st.spinner("🔄 Loading Neural Style Detector (BERT)..."):
            tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            mdl.eval().to(DEVICE)
        return tok, mdl
    except Exception as e:
        st.error(f"BERT Load Error: {e}")
        return None, None

@st.cache_resource
def load_tfidf_model():
    """Attempts to load local TF-IDF. Returns None if missing."""
    path = MODELS / "tfidf_logreg_improved.joblib"
    if path.exists():
        return joblib.load(path)
    path_orig = MODELS / "tfidf_logreg.joblib"
    if path_orig.exists():
        return joblib.load(path_orig)
    return None

@st.cache_resource
def load_nli_model():
    with st.spinner("🔄 Loading Fact-Checking Engine (DeBERTa)..."):
        return CrossEncoder('cross-encoder/nli-deberta-v3-small')

# --- 2) PREDICTION ENGINES ---

def analyze_bert(text, tok, mdl):
    """Returns prediction, probability, and attention scores."""
    encoded = tok(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=CHUNK_OVERLAP,
        return_overflowing_tokens=True,
        padding=False
    )

    chunk_logits, attentions, input_ids_list = [], [], []

    with torch.no_grad():
        for i in range(len(encoded["input_ids"])):
            inp_ids = torch.tensor([encoded["input_ids"][i]]).to(DEVICE)
            mask = torch.tensor([encoded["attention_mask"][i]]).to(DEVICE)
            try:
                out = mdl(input_ids=inp_ids, attention_mask=mask, output_attentions=True)
                chunk_logits.append(out.logits[0].cpu().numpy())
                attn = torch.stack(out.attentions)[-1].mean(dim=1)[0]  # last layer, avg heads
                token_importance = attn.sum(dim=0).cpu().numpy()
                attentions.append(token_importance)
            except Exception:
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
        "prob_fake": float(probs[1]) if probs.shape[0] > 1 else 0.0,
        "tokens": tokens,
        "attention": attentions[0] if attentions else np.zeros(len(tokens))
    }

def analyze_tfidf(text, pipeline):
    if pipeline is None:
        return None
    try:
        probs = pipeline.predict_proba([text])[0]
        pred = int(np.argmax(probs))
        vectorizer = pipeline.named_steps.get('tfidf', pipeline.named_steps.get('vectorizer'))
        clf = pipeline.named_steps.get('clf', pipeline.named_steps.get('logreg'))
        if (vectorizer is None) or (clf is None):
            return {"pred": pred, "prob_fake": float(probs[1]), "features": []}
        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]
        response = vectorizer.transform([text]).tocoo()
        feats = [(feature_names[col], val * coefs[col]) for col, val in zip(response.col, response.data)]
        feats.sort(key=lambda x: abs(x[1]), reverse=True)
        return {"pred": pred, "prob_fake": float(probs[1]), "features": feats[:25]}
    except Exception as e:
        st.warning(f"TF-IDF analysis error: {e}")
        return None

# ---------- FACT CHECKING HELPERS ----------

CATEGORY_SITES = {
    "General": (
        "reuters.com","apnews.com","bbc.com","bbc.co.uk","ft.com","nytimes.com",
        "theguardian.com","npr.org","washingtonpost.com","aljazeera.com","euronews.com"
    ),
    "Politics": (
        "reuters.com","apnews.com","bbc.com","bbc.co.uk","ft.com","nytimes.com",
        "theguardian.com","politico.com","npr.org","washingtonpost.com","aljazeera.com",
        "euronews.com","congress.gov","whitehouse.gov","europa.eu","europarl.europa.eu"
    ),
    "Business": (
        "reuters.com","bloomberg.com","ft.com","wsj.com","cnbc.com","apnews.com",
        "sec.gov","investorrelations","ir."  # simple IR heuristics
    ),
    "Tech": (
        "reuters.com","theverge.com","wired.com","arstechnica.com","techcrunch.com",
        "ft.com","wsj.com","bbc.com"
    ),
    "Science/Health": (
        "nature.com","science.org","nih.gov","who.int","cdc.gov","thelancet.com",
        "nejm.org","reuters.com","bbc.com","apnews.com"
    ),
    "Sports": (
        "espn.com","bbc.com/sport","skysports.com","uefa.com","fifa.com","mlb.com",
        "nba.com","nfl.com","nhl.com","olympics.com","reuters.com"
    ),
    "Climate/Environment": (
        "ipcc.ch","noaa.gov","nasa.gov","un.org","nature.com","science.org",
        "reuters.com","bbc.com","apnews.com","climate.nasa.gov"
    ),
    "War/Conflict": (
        "reuters.com","apnews.com","bbc.com","aljazeera.com","theguardian.com",
        "nytimes.com","ft.com","icrc.org","euronews.com"
    ),
}

REGION_OPTIONS = ["Auto","us-en","uk-en","it-it","de-de","fr-fr","es-es","in-en","au-en"]
RECENCY_OPTIONS = {"Any time": None, "Past day": "d", "Past week": "w", "Past month": "m", "Past year": "y"}

def build_queries_from_claim(claim: str, category: str):
    """Construct multiple high-signal queries preserving entities and adding site filters."""
    quoted = re.findall(r'"([^"]+)"', claim)
    caps_spans = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b', claim)
    entities = []
    for s in quoted + caps_spans:
        if s not in entities:
            entities.append(s)

    claim_l = claim.lower().replace("ageing", "aging")
    number_terms = re.findall(r'\b\d{2,3}\b', claim_l)[:4]

    # topical terms by category
    topic_terms = {
        "General": [],
        "Politics": ["election","policy","bill","parliament","government","president","minister"],
        "Business": ["earnings","revenue","merger","acquisition","IPO","guidance"],
        "Tech": ["launch","update","AI","software","chip","cybersecurity"],
        "Science/Health": ["study","trial","peer-reviewed","research","meta-analysis","health"],
        "Sports": ["match","fixture","transfer","tournament","final"],
        "Climate/Environment": ["emissions","climate","warming","renewable","carbon"],
        "War/Conflict": ["offensive","ceasefire","sanctions","casualties","aid"]
    }[category]

    # base queries
    queries = []
    if entities:
        queries.append(f'"{entities[0]}" {" ".join(topic_terms)} {" ".join(number_terms)}'.strip())
        if len(entities) > 1:
            queries.append(f'"{entities[0]}" "{entities[1]}" {" ".join(number_terms)}'.strip())
    else:
        # fallback to content words if no entities
        words = [w for w in re.sub(r'[^\w\s]',' ',claim).split() if len(w) > 2]
        queries.append(" ".join(words[:12]))

    # category-focused site filters
    for site in CATEGORY_SITES.get(category, ()):
        base = f'"{entities[0]}"' if entities else " ".join(topic_terms[:2]) or "news"
        queries.append(f'site:{site} {base}')

    # dedupe
    seen, out = set(), []
    for q in queries:
        q = q.strip()
        if q and (q not in seen):
            out.append(q)
            seen.add(q)
    return out[:6], [e.lower() for e in entities[:3]]  # return a few entity keys

def perform_search(q, mode="text", region="us-en", timelimit=None):
    """DuckDuckGo search with optional region & recency (timelimit: d/w/m/y)."""
    results = []
    try:
        with DDGS() as ddgs:
            kwargs = dict(region=region, safesearch='moderate', max_results=8)
            if timelimit:
                kwargs["timelimit"] = timelimit
            if mode == "news":
                ddg_results = list(ddgs.news(q, **kwargs))
                for r in ddg_results:
                    text_content = r.get('body', r.get('snippet', ''))
                    if text_content and len(text_content) > 50:
                        results.append({
                            "text": text_content,
                            "url": r.get('link', r.get('href', r.get('url', ''))),
                            "title": r.get('title', 'Untitled'),
                            "source_name": r.get('source', 'News')
                        })
            else:
                ddg_results = list(ddgs.text(q, **kwargs))
                for r in ddg_results:
                    text_content = r.get('body', r.get('snippet', ''))
                    if text_content and len(text_content) > 50:
                        results.append({
                            "text": text_content,
                            "url": r.get('href', r.get('link', r.get('url', ''))),
                            "title": r.get('title', 'Untitled'),
                            "source_name": r.get('source', 'Web')
                        })
    except Exception as e:
        print(f"DDG {mode} search error: {e}")
    return results

def is_mostly_english(text: str) -> bool:
    if not text:
        return False
    latin_chars = sum(1 for c in text if ord(c) < 128)
    return (latin_chars / max(1, len(text))) > 0.8

def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def verify_facts(text, nli_model, category="General", region="us-en", timelimit=None, english_only=False, min_unique_domains=2):
    """
    Robust fact-checker across categories with entity gating, domain preference, recency & region controls,
    calibrated NLI, and source-diversity safeguard.
    """
    # 1) Claim extraction
    sentences = [s.strip() for s in re.split(r'[.!?]', text.strip()) if s.strip()]
    def score_sentence(sent):
        score = 0; words = sent.split()
        if 8 <= len(words) <= 25: score += 3
        elif len(words) > 25: score += 1
        proper_nouns = sum(1 for i, w in enumerate(words) if i > 0 and w and w[0].isupper()); score += proper_nouns * 2
        claim_verbs = ['announced','reported','confirmed','said','stated','claimed','revealed','discovered','found','declared','admitted','denied','occurred','caused','led','resulted','approved','passed','signed']
        if any(verb in sent.lower() for verb in claim_verbs): score += 3
        if re.search(r'\d+', sent): score += 2
        if '"' in sent or "'" in sent: score += 1
        return score
    scored = sorted(((s, score_sentence(s)) for s in sentences[:5]), key=lambda x: x[1], reverse=True)
    claim = scored[0][0] if scored else text[:300]
    if scored and scored[0][1] < 3 and len(scored) > 1:
        claim = " ".join([s for s, _ in scored[:2]])


    # 2) Queries + entity keys
    queries, entity_keys = build_queries_from_claim(claim, category)

    # 3) Search (news first), filter, domain preference
    raw_results = []
    for i, q in enumerate(queries):
        mode = "news" if i == 0 else "text"
        raw_results.extend(perform_search(q, mode=mode, region=region, timelimit=timelimit))

    GOOD = CATEGORY_SITES.get(category, CATEGORY_SITES["General"])
    good_tuple = tuple(GOOD)

    seen_urls, evidence, sources_found = set(), [], []
    for res in raw_results:
        text_content = res.get('text', '')
        title = res.get('title', '')
        url = res.get('url', '#')
        if (not text_content) or (len(text_content) < 20) or (not title) or (url in seen_urls):
            continue
        if english_only and not (is_mostly_english(text_content) and is_mostly_english(title)):
            continue

        hay = (title + " " + text_content).lower()
        # Filtro entità meno restrittivo: se non trovi nulla, mostra comunque i risultati generici
        if entity_keys and not any(k in hay for k in entity_keys):
            continue

        seen_urls.add(url)
        d = domain_of(url)
        icon = "🔗 Web"
        if "bbc" in d: icon = "🇬🇧 BBC"
        elif "bloomberg" in d: icon = "📈 Bloomberg"
        elif "reuters" in d: icon = "📰 Reuters"
        elif "cnn" in d: icon = "🇺🇸 CNN"
        elif "nytimes" in d: icon = "🗞️ NYT"
        elif "apnews" in d or "ap.org" in d: icon = "📡 AP"
        elif "theguardian" in d: icon = "🇬🇧 Guardian"
        elif "washingtonpost" in d: icon = "📰 WaPo"
        elif "ft.com" in d: icon = "💼 FT"

        domain_bonus = any(s in d for s in good_tuple) or d.startswith("ir.") or "investorrelations" in d

        evidence.append(text_content)
        sources_found.append({
            "text": text_content,
            "url": url,
            "source": icon,
            "title": title,
            "domain": d,
            "good_domain": domain_bonus
        })

    # Fallback: se non trovi nulla con entità, riprova con una query generica
    if not evidence:
        fallback_query = claim if len(claim.split()) < 12 else " ".join(claim.split()[:12])
        fallback_results = perform_search(fallback_query, mode="news", region=region, timelimit=timelimit)
        for res in fallback_results:
            text_content = res.get('text', '')
            title = res.get('title', '')
            url = res.get('url', '#')
            if (not text_content) or (len(text_content) < 20) or (not title) or (url in seen_urls):
                continue
            seen_urls.add(url)
            d = domain_of(url)
            icon = "🔗 Web"
            if "bbc" in d: icon = "🇬🇧 BBC"
            elif "bloomberg" in d: icon = "📈 Bloomberg"
            elif "reuters" in d: icon = "📰 Reuters"
            elif "cnn" in d: icon = "🇺🇸 CNN"
            elif "nytimes" in d: icon = "🗞️ NYT"
            elif "apnews" in d or "ap.org" in d: icon = "📡 AP"
            elif "theguardian" in d: icon = "🇬🇧 Guardian"
            elif "washingtonpost" in d: icon = "📰 WaPo"
            elif "ft.com" in d: icon = "💼 FT"
            domain_bonus = any(s in d for s in good_tuple) or d.startswith("ir.") or "investorrelations" in d
            evidence.append(text_content)
            sources_found.append({
                "text": text_content,
                "url": url,
                "source": icon,
                "title": title,
                "domain": d,
                "good_domain": domain_bonus
            })
    if not evidence:
        return {
            "status": "NO_DATA",
            "claim": claim,
            "queries": queries,
            "verdict": "UNVERIFIED",
            "evidence": [],
            "confidence": 0.0,
            "category": category,
            "region": region,
            "recency": timelimit or "any"
        }

    # Prefer good domains, then longer snippets
    order = sorted(
        range(len(sources_found)),
        key=lambda i: (not sources_found[i]["good_domain"], -len(sources_found[i]["text"]))
    )
    evidence = [evidence[i] for i in order]
    sources_found = [sources_found[i] for i in order]

    # 4) NLI Verification (calibrated)
    try:
        pairs = [(claim, e) for e in evidence]
        logits = nli_model.predict(pairs)  # [contradiction, entailment, neutral]
        probs = softmax_rows(np.asarray(logits))
        p_contra, p_entail, p_neutral = probs[:,0], probs[:,1], probs[:,2]
        max_entail, max_contra, avg_entail = float(p_entail.max()), float(p_contra.max()), float(p_entail.mean())

        if max_entail >= 0.80:
            verdict, confidence = "VERIFIED", max_entail
        elif max_contra >= 0.70:
            verdict, confidence = "DEBUNKED", max_contra
        elif avg_entail >= 0.55 or (max_entail >= 0.65 and avg_entail >= 0.45):
            verdict, confidence = "LIKELY TRUE", avg_entail
        elif max_contra >= 0.50:
            verdict, confidence = "QUESTIONABLE", max_contra
        else:
            verdict, confidence = "INCONCLUSIVE", max(max_entail, max_contra)

        # Source diversity safeguard (helps for politics/business rumor mills)
        unique_domains = {s["domain"] for s in sources_found if s.get("domain")}
        if verdict == "VERIFIED" and len(unique_domains) < min_unique_domains:
            verdict = "LIKELY TRUE"  # downgrade if all support comes from too few domains

        for i, src in enumerate(sources_found):
            src_probs = probs[i].tolist()
            src['probs'] = src_probs
            label_idx = int(np.argmax(probs[i]))
            src['label'] = ['CONTRADICTS', 'SUPPORTS', 'NEUTRAL'][label_idx]

        return {
            "status": "SUCCESS",
            "claim": claim,
            "queries": queries,
            "verdict": verdict,
            "confidence": float(confidence),
            "evidence": sources_found,
            "category": category,
            "region": region,
            "recency": timelimit or "any",
            "unique_domains": len(unique_domains)
        }
    except Exception as e:
        print(f"NLI processing error: {e}")
        return {
            "status": "ERROR",
            "claim": claim,
            "queries": queries,
            "verdict": "UNVERIFIED",
            "evidence": sources_found,
            "confidence": 0.0,
            "category": category,
            "region": region,
            "recency": timelimit or "any"
        }

# --- 3) VISUALIZATION ---

def plot_attention_bar(tokens, scores):
    import string
    punct = set(string.punctuation)
    clean = [(t, s) for t, s in zip(tokens, scores)
             if t not in ['[CLS]', '[SEP]', '[PAD]'] and t not in punct]
    if not clean:
        return plt.figure()
    c_tokens, c_scores = zip(*clean)
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(set(c_scores)) == 1:
        norm_scores = np.zeros_like(c_scores, dtype=float)
    else:
        mn, mx = float(min(c_scores)), float(max(c_scores))
        norm_scores = (np.array(c_scores) - mn) / max(1e-9, (mx - mn))
    colors = Blues(norm_scores)
    display_count = min(30, len(c_tokens))
    ax.bar(range(display_count), c_scores[:display_count], color=colors[:display_count])
    ax.set_xticks(range(display_count))
    ax.set_xticklabels(c_tokens[:display_count], rotation=45, ha='right', fontsize=9)
    ax.set_title("Neural Attention Focus", fontsize=12, fontweight='bold')
    ax.set_ylabel("Attention Weight")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_tfidf_importance(features):
    if not features:
        return plt.figure()
    display_count = min(20, len(features))
    words = [f[0] for f in features[:display_count]][::-1]
    scores = [f[1] for f in features[:display_count]][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d32f2f' if s > 0 else '#388e3c' for s in scores]
    ax.barh(words, scores, color=colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title("Keyword Forensic Weights (Red=Fake, Green=Real)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature Weight")
    plt.tight_layout()
    return fig

# --- 4) UI ---

st.title("🛡️ TruthLens: Forensic News Dashboard")
st.markdown("*Neural style + Keyword forensics + Real-time fact-checking, now tuned for politics, business, tech, science, sports, climate, and conflict.*")

# Controls (sidebar)
with st.sidebar:
    st.header("Search Controls")
    category = st.selectbox("News category", list(CATEGORY_SITES.keys()), index=0)
    region_sel = st.selectbox("Region / language", REGION_OPTIONS, index=0)
    recency_sel = st.selectbox("Recency", list(RECENCY_OPTIONS.keys()), index=0)
    english_only = st.toggle("English sources only", value=True)
    min_unique_domains = st.slider("Min unique domains to verify", 1, 5, 2, help="Require at least this many distinct sources before calling VERIFIED.")
    st.caption("Tip: tighten recency for breaking stories; widen for backgrounders.")

# Main input
text = st.text_area(
    "Input Article Text or Claim",
    height=160,
    placeholder="Paste the article, quote, or claim you want to check...",
    label_visibility="visible"
)

if st.button("🚀 RUN FULL FORENSIC ANALYSIS", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please enter text to analyze.")
    else:
        # Load models
        bert_tok, bert_mdl = load_bert_model()
        tfidf_pipe = load_tfidf_model()
        nli_mdl = load_nli_model()
        if bert_tok is None or bert_mdl is None:
            st.error("Failed to load BERT model. Please check model configuration.")
            st.stop()

        with st.spinner("🕵️‍♀️ Running Multi-Layer Analysis..."):
            bert_res = analyze_bert(text, bert_tok, bert_mdl)
            tfidf_res = analyze_tfidf(text, tfidf_pipe)
            region = "us-en" if region_sel == "Auto" else region_sel
            timelimit = RECENCY_OPTIONS[recency_sel]
            fact_res = verify_facts(
                text, nli_mdl,
                category=category, region=region, timelimit=timelimit,
                english_only=english_only, min_unique_domains=min_unique_domains
            )
            bert_lbl = "FAKE" if bert_res['pred'] == 1 else "REAL"
            tfidf_lbl = ("FAKE" if (tfidf_res and tfidf_res['pred'] == 1)
                         else ("REAL" if tfidf_res else "N/A"))

        st.divider()

        # VERDICT BOX with improved logic
        final_color = "#f0f0f0"; final_msg = "ANALYSIS COMPLETE"
        if fact_res['verdict'] == "DEBUNKED":
            final_color = "#ffebee"; final_msg = "🔴 **DEBUNKED**: External sources contradict this claim"
        elif fact_res['verdict'] == "VERIFIED":
            final_color = "#e8f5e9"; final_msg = "🟢 **VERIFIED**: Confirmed by multiple reliable sources"
        elif fact_res['verdict'] == "LIKELY TRUE":
            final_color = "#e8f5e9"; final_msg = "🟢 **LIKELY TRUE**: Supported by credible sources"
        elif fact_res['verdict'] == "QUESTIONABLE":
            final_color = "#fff3e0"; final_msg = "🟠 **QUESTIONABLE**: Conflicting or weak evidence"
        elif bert_lbl == "FAKE" and bert_res['prob_fake'] > 0.7:
            final_color = "#ffebee"; final_msg = "🔴 **SUSPICIOUS**: High probability of deceptive writing style"
        elif bert_lbl == "FAKE":
            final_color = "#fff3e0"; final_msg = "🟠 **CAUTION**: Possible fake news patterns detected"
        else:
            final_color = "#e3f2fd"; final_msg = "🔵 **ANALYSIS COMPLETE**: Review results below"

        st.markdown(f"""
        <div style="background-color: {final_color}; padding: 20px; border-radius: 10px; 
                    text-align: center; margin-bottom: 20px; border: 2px solid rgba(0,0,0,0.1);">
            <h2 style="margin:0; color: #1e1e1e;">{final_msg}</h2>
            <div class="small">Category: <b>{fact_res.get('category','—')}</b> • Region: <b>{fact_res.get('region','—')}</b> • Recency: <b>{fact_res.get('recency','any')}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # METRICS
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🧠 Neural Style")
            st.metric("Verdict", bert_lbl, f"{bert_res['prob_fake']:.1%} fake-prob",
                      delta_color="inverse" if bert_lbl == "FAKE" else "normal")
        with c2:
            st.markdown("### 🧮 Keyword Analysis")
            if tfidf_res:
                st.metric("Verdict", tfidf_lbl, f"{tfidf_res['prob_fake']:.1%} fake-prob",
                          delta_color="inverse" if tfidf_lbl == "FAKE" else "normal")
            else:
                st.metric("Verdict", "N/A", "Model not available")
        with c3:
            st.markdown("### 🌍 Fact Check")
            v = fact_res['verdict']
            conf_str = f"{fact_res.get('confidence', 0):.1%}" if fact_res.get('confidence') else "—"
            color = "normal" if v in ["VERIFIED", "LIKELY TRUE"] else "inverse" if v == "DEBUNKED" else "off"
            st.metric("Result", v, conf_str, delta_color=color)

        # TABS
        st.markdown("---")
        t1, t2, t3 = st.tabs(["🧠 Neural Attention", "🧮 Keyword Forensics", "🌍 Evidence Locker"])

        with t1:
            st.markdown("**Attention Heatmap**: Shows which words the neural network focused on")
            fig = plot_attention_bar(bert_res['tokens'], bert_res['attention'])
            st.pyplot(fig)
            st.caption("Higher bars = words that influenced the model's decision more")

        with t2:
            if tfidf_res and tfidf_res.get('features'):
                st.markdown("**Feature Importance**: Keywords associated with fake (red) or real (green) news")
                fig = plot_tfidf_importance(tfidf_res['features'])
                st.pyplot(fig)
                st.caption("Based on statistical patterns in training data")
            else:
                st.info("TF-IDF model not available or no significant features found")

        with t3:
            if fact_res.get('claim'):
                st.markdown("**📋 Claim Being Verified:**")
                st.info(f'"{fact_res["claim"]}"')
                st.markdown("---")

            tried = fact_res.get('queries', [])
            if tried:
                st.markdown("**🔍 Queries tried:**")
                for q in tried[:3]:
                    st.code(q)
                st.markdown("---")

            st.caption(f"Status: {fact_res['status']} • Sources: {len(fact_res.get('evidence', []))} • Unique domains: {fact_res.get('unique_domains','—')}")

            if fact_res['status'] == "SUCCESS" and fact_res.get('evidence'):
                st.markdown("---")
                for idx, item in enumerate(fact_res['evidence'], 1):
                    probs = item.get('probs', [0.33, 0.33, 0.34])
                    contradiction, entailment, neutral = probs
                    label = item.get('label', 'NEUTRAL')
                    if label == 'SUPPORTS':
                        emoji, color, val = "✅", "green", entailment
                    elif label == 'CONTRADICTS':
                        emoji, color, val = "❌", "red", contradiction
                    else:
                        emoji, color, val = "⚖️", "orange", neutral

                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{item['source']}** — [{item['title']}]({item['url']})")
                        with col_b:
                            st.markdown(f":{color}[**{emoji} {label} ({val:.0%})**]")
                        st.caption(f"\"{item['text'][:350]}...\"")
                        if idx < len(fact_res['evidence']):
                            st.divider()
            elif fact_res['status'] == "NO_DATA":
                st.warning("⚠️ No external sources found. Unable to fact-check this claim.")
                st.info("💡 Try adjusting category/region, widening recency, or adding specific names/places.")
            else:
                st.error("❌ Error during fact-checking process. Please try again.")

else:
    st.info("👆 Paste an article or claim, choose a category (e.g., Politics or Business), then run the analysis.")
    with st.expander("ℹ️ How TruthLens Works"):
        st.markdown("""
        **Three-Layer Detection System:**
        1) **🧠 Neural Style Analysis** — BERT flags deceptive linguistic patterns  
        2) **🧮 Keyword Forensics** — TF-IDF weights words correlated with fake/real  
        3) **🌍 Real-Time Fact Checking** — Category-aware search + NLI verification  
        """)
