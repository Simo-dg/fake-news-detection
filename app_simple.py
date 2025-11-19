import os
from pathlib import Path
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).parent.resolve()
MODELS = BASE / "models"

MAX_LENGTH = 512
CHUNK_OVERLAP = 128

st.set_page_config(page_title="📰 Fake News Detector", layout="wide")
st.title("📰 Fake News Detector - Interpretability")

st.warning("""
⚠️ **LIMITAZIONI IMPORTANTI**:
- I modelli rilevano lo **STILE** (formale vs clickbait), non la **VERITÀ** dei fatti
- Addestrati su Reuters (REAL) vs articoli sensazionalistici (FAKE)
- **Non fanno fact-checking** - possono essere ingannati da fake news ben scritte
- La verifica Wikipedia è un controllo aggiuntivo ma non definitivo

💡 **Usare sempre fact-checking professionale per contenuti importanti!**
""")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sidebar - Model selection
available_models = {}
if (MODELS/"bert_finetuned").exists():
    available_models["BERT (fine-tuned)"] = "bert"
if (MODELS/"tfidf_logreg_improved.joblib").exists():
    available_models["TF-IDF Improved (with preprocessing)"] = "tfidf_improved"
if (MODELS/"tfidf_logreg.joblib").exists():
    available_models["TF-IDF Original (no preprocessing)"] = "tfidf"

if not available_models:
    st.error("Nessun modello trovato! Assicurati di aver addestrato almeno un modello.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Seleziona modello:",
    list(available_models.keys()),
    help="BERT è un modello deep learning (black box), TF-IDF è interpretabile tramite feature importance"
)
model_type = available_models[model_choice]

@st.cache_resource
def load_bert_model():
    model_id = "Simingasa/fake-news-bert-finetuned"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    mdl.to(DEVICE)
    return tok, mdl

@st.cache_resource
def load_tfidf_model(model_name="improved"):
    if model_name == "improved":
        return joblib.load(MODELS/"tfidf_logreg_improved.joblib")
    else:
        return joblib.load(MODELS/"tfidf_logreg.joblib")

def predict_with_chunks(text, tok, mdl):
    """Predizione BERT con chunking come nel training."""
    encoded = tok(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=CHUNK_OVERLAP,
        return_overflowing_tokens=True,
        padding=False
    )
    
    chunk_logits = []
    num_chunks = len(encoded["input_ids"])
    
    with torch.no_grad():
        for i in range(num_chunks):
            chunk_input = {
                "input_ids": torch.tensor([encoded["input_ids"][i]]).to(DEVICE),
                "attention_mask": torch.tensor([encoded["attention_mask"][i]]).to(DEVICE)
            }
            out = mdl(**chunk_input).logits[0]
            chunk_logits.append(out.cpu().numpy())
    
    doc_logits = np.mean(chunk_logits, axis=0)
    doc_probs = np.exp(doc_logits) / np.sum(np.exp(doc_logits))
    doc_pred = np.argmax(doc_logits)
    
    return {
        "prediction": int(doc_pred),
        "probabilities": {"REAL": float(doc_probs[0]), "FAKE": float(doc_probs[1])},
        "num_chunks": num_chunks
    }

def predict_tfidf(text, pipeline):
    """Predizione TF-IDF + Logistic Regression."""
    proba = pipeline.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    
    return {
        "prediction": pred,
        "probabilities": {"REAL": float(proba[0]), "FAKE": float(proba[1])},
        "num_chunks": 1
    }

def get_attention_for_chunk(text, tok, mdl, chunk_idx=0):
    """Ottiene l'attention per un chunk specifico."""
    encoded = tok(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=CHUNK_OVERLAP,
        return_overflowing_tokens=True,
        padding=False
    )
    
    if chunk_idx >= len(encoded["input_ids"]):
        chunk_idx = 0
    
    chunk_input = {
        "input_ids": torch.tensor([encoded["input_ids"][chunk_idx]]).to(DEVICE),
        "attention_mask": torch.tensor([encoded["attention_mask"][chunk_idx]]).to(DEVICE)
    }
    
    tokens = tok.convert_ids_to_tokens(encoded["input_ids"][chunk_idx])
    
    with torch.no_grad():
        out = mdl(**chunk_input, output_attentions=True)
    
    attn = torch.stack(out.attentions)
    attn_avg = attn.mean(dim=2).mean(dim=0)[0].cpu().numpy()
    
    return tokens, attn_avg

def get_tfidf_feature_importance(text, pipeline):
    """Ottiene l'importanza delle feature TF-IDF."""
    vectorizer = None
    classifier = None
    
    for name, step in pipeline.steps:
        if 'tfidf' in name.lower() or 'vectorizer' in name.lower():
            vectorizer = step
        elif 'logreg' in name.lower() or 'logistic' in name.lower() or 'clf' in name.lower():
            classifier = step
    
    if vectorizer is None or classifier is None:
        return [], []
    
    X = vectorizer.transform([text])
    
    # Per logistic regression binaria, c'è solo coef_[0]
    # Positivo -> classe 1 (FAKE), Negativo -> classe 0 (REAL)
    coefs = classifier.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    
    X_coo = X.tocoo()
    present_features = []
    
    for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
        feature_name = feature_names[j]
        contribution = float(v * coefs[j])
        present_features.append((feature_name, contribution))
    
    present_features.sort(key=lambda x: abs(x[1]), reverse=True)
    top_n = min(30, len(present_features))
    top_features = present_features[:top_n]
    
    if not top_features:
        return [], []
    
    words = [f[0] for f in top_features]
    scores = np.array([f[1] for f in top_features])
    
    return words, scores

def plot_attention_for_token(tokens, attention_matrix, token_idx, hide_special=False, hide_punct=False):
    """Visualizza l'attention di un token."""
    attention_scores = attention_matrix[token_idx, :]
    
    if hide_special or hide_punct:
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        punctuation = [',', '.', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '-', '–', '—']
        
        keep_indices = []
        for i, t in enumerate(tokens):
            skip = False
            if hide_special and t in special_tokens:
                skip = True
            if hide_punct and t in punctuation:
                skip = True
            if not skip:
                keep_indices.append(i)
        
        if len(keep_indices) == 0:
            keep_indices = list(range(len(tokens)))
        
        filtered_tokens = [tokens[i] for i in keep_indices]
        filtered_scores = attention_scores[keep_indices]
        
        if filtered_scores.sum() > 0:
            filtered_scores = filtered_scores / filtered_scores.sum()
    else:
        filtered_tokens = tokens
        filtered_scores = attention_scores
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(filtered_tokens) * 0.25)))
    
    from matplotlib.cm import YlOrRd
    norm_scores = (filtered_scores - filtered_scores.min()) / (filtered_scores.max() - filtered_scores.min() + 1e-8)
    colors = [YlOrRd(s) for s in norm_scores]
    
    y_pos = np.arange(len(filtered_tokens))
    ax.barh(y_pos, filtered_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filtered_tokens, fontsize=9)
    ax.set_xlabel('Attention Weight', fontsize=10)
    
    title = f"Attention from '{tokens[token_idx]}' (position {token_idx})"
    filters = []
    if hide_special:
        filters.append("no special tokens")
    if hide_punct:
        filters.append("no punctuation")
    if filters:
        title += f" - {', '.join(filters)}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_feature_importance(words, scores, pred_label):
    """Visualizza l'importanza delle feature TF-IDF."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(words) * 0.3)))
    
    # Verde per FAKE (positivo), Rosso per REAL (negativo)
    colors = ['#d32f2f' if s > 0 else '#388e3c' for s in scores]
    
    y_pos = np.arange(len(words))
    ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel('Contributo (TF-IDF × Coeff)', fontsize=10)
    ax.set_title(f"Feature Importance - Predizione: {pred_label}", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.invert_yaxis()
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d32f2f', alpha=0.7, label='→ FAKE (positivo)'),
        Patch(facecolor='#388e3c', alpha=0.7, label='→ REAL (negativo)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    return fig

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ Differenze tra i modelli:

**BERT (fine-tuned)**
- Deep learning model (black box)
- Visualizzazione: Attention patterns
- Mostra dove il modello "guarda"
- Difficile interpretare il perché

**TF-IDF Improved**
- Modello lineare interpretabile
- Con preprocessing (lowercase, stop words)
- Feature importance semantiche
- Robusto e trasparente

**TF-IDF Original**
- Senza preprocessing
- Impara anche pattern stilistici
- "US" maiuscolo diverso da "us"
- Utile per vedere bias del dataset
""")

# Main UI
text = st.text_area("Inserisci un articolo:", height=200, placeholder="Scrivi o incolla il testo dell'articolo...")

if st.button("Analizza", type="primary"):
    if not text.strip():
        st.warning("Inserisci del testo prima di analizzare.")
    else:
        with st.spinner("Analyzing..."):
            if model_type == "bert":
                tok, mdl = load_bert_model()
                result = predict_with_chunks(text, tok, mdl)
            elif model_type == "tfidf_improved":
                pipeline = load_tfidf_model("improved")
                result = predict_tfidf(text, pipeline)
            else:  # tfidf original
                pipeline = load_tfidf_model("original")
                result = predict_tfidf(text, pipeline)
            
            # Mostra risultati
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_label = "🔴 FAKE" if result["prediction"] == 1 else "🟢 REAL"
                st.metric("Predizione", pred_label)
            
            with col2:
                prob_fake = result["probabilities"]["FAKE"]
                st.metric("Probabilità FAKE", f"{prob_fake:.2%}")
            
            with col3:
                if model_type == "bert":
                    st.metric("Chunks processati", result["num_chunks"])
                else:
                    st.metric("Modello", "TF-IDF")
            
            with st.expander("📊 Dettagli probabilità"):
                st.json(result["probabilities"])
            
            st.divider()
            # Interpretability visualization
            if model_type == "bert":
                st.subheader("🔍 Attention Visualization (BERT)")
                st.caption("⚠️ BERT è un modello black box. L'attention mostra dove guarda il modello, ma non spiega completamente il perché delle decisioni.")
                
                if result["num_chunks"] > 1:
                    chunk_idx = st.slider("Seleziona il chunk da analizzare:", 0, result["num_chunks"] - 1, 0)
                else:
                    chunk_idx = 0
                
                tokens, attn_matrix = get_attention_for_chunk(text, tok, mdl, chunk_idx)
                
                st.write(f"**Chunk {chunk_idx + 1} di {result['num_chunks']}** - {len(tokens)} tokens")
                
                col1, col2 = st.columns(2)
                with col1:
                    hide_special = st.checkbox(
                        "Nascondi token speciali ([CLS], [SEP], [PAD])",
                        value=True,
                        help="I token speciali spesso ricevono molta attention come 'hub' o 'parcheggio'."
                    )
                with col2:
                    hide_punct = st.checkbox(
                        "Nascondi punteggiatura (, . : ; ecc.)",
                        value=True,
                        help="La punteggiatura spesso riceve alta attention per ragioni sintattiche, non semantiche."
                    )
                
                token_idx = st.selectbox(
                    "Seleziona un token per vedere dove concentra l'attention:",
                    range(len(tokens)),
                    format_func=lambda i: f"{i}: {tokens[i]}"
                )
                
                if not hide_special or not hide_punct:
                    with st.expander("ℹ️ Perché nascondere token speciali e punteggiatura?"):
                        st.markdown("""
                        **Token speciali ([CLS], [SEP], [PAD])**:
                        - Vengono usati come "hub" per attention non utile
                        - [CLS] aggrega informazione per la classificazione
                        - [SEP] marca la fine della sequenza
                        - Spesso hanno attention molto alta ma non sono semanticamente rilevanti
                        
                        **Punteggiatura (, . : ; " ' ecc.)**:
                        - Serve come "parcheggio" per attention patterns
                        - Marca confini sintattici (frasi, clausole)
                        - Alta attention per ragioni strutturali, non di contenuto
                        - Il modello impara che delimitano unità sintattiche
                        
                        💡 Nascondendo entrambi vedi meglio le relazioni semantiche tra le **parole di contenuto**!
                        """)
                
                fig = plot_attention_for_token(tokens, attn_matrix, token_idx, hide_special, hide_punct)
                st.pyplot(fig)
                
            else:  # TF-IDF (both improved and original)
                st.subheader("📊 Feature Importance (TF-IDF)")
                
                if model_type == "tfidf_improved":
                    st.caption("✅ TF-IDF con preprocessing: lowercase, stop words removal, normalizzazione. Più robusto e interpretabile.")
                else:
                    st.caption("⚠️ TF-IDF senza preprocessing: sensibile a maiuscole/minuscole e parole comuni. Può apprendere pattern stilistici invece che semantici.")
                
                words, scores = get_tfidf_feature_importance(text, pipeline)
                
                if len(words) > 0:
                    pred_label = "FAKE" if result["prediction"] == 1 else "REAL"
                    fig = plot_feature_importance(words, scores, pred_label)
                    st.pyplot(fig)
                    
                    with st.expander("ℹ️ Come interpretare i risultati"):
                        st.markdown("""
                        **Barre rosse (positive)**: Parole che spingono verso FAKE
                        
                        **Barre verdi (negative)**: Parole che spingono verso REAL
                        
                        **Calcolo**: TF-IDF value × Coefficiente Logistic Regression
                        
                        In Logistic Regression binaria:
                        - Coefficienti positivi → aumentano probabilità di FAKE (classe 1)
                        - Coefficienti negativi → aumentano probabilità di REAL (classe 0)
                        
                        💡 Le parole più in alto hanno il maggiore impatto assoluto sulla decisione!
                        """)
                else:
                    st.warning("Nessuna feature rilevante trovata nel testo.")

# Footer
st.divider()
if model_type == "bert":
    st.caption("💡 Il modello BERT processa testi lunghi dividendoli in chunks di 512 token con overlap di 128 token.")
else:
    st.caption("💡 TF-IDF rappresenta il testo come frequenze pesate delle parole, permettendo interpretabilità completa.")
