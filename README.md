# 🛡️ TruthLens: Forensic Fake News Detection System

**TruthLens** is a multi-layered news verification platform that combines transformer-based deep learning (BERT), heuristic analysis (TF-IDF), topic modeling (BERTopic), and live cross-referencing (NLI-ready interface) to detect misinformation.

Unlike standard classifiers, TruthLens focuses on **forensic text analysis**—cleaning specific “data artifacts” (e.g., “(Reuters)” headers or “Click here” buttons) so models learn **semantic untruths** rather than platform-specific metadata.

<p align="center">
  <img alt="TruthLens" src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  <img alt="Framework" src="https://img.shields.io/badge/ML-BERT%2C%20TF--IDF%2C%20BERTopic-orange" />
  <img alt="UI" src="https://img.shields.io/badge/UI-Streamlit-red" />
</p>

---

## ✨ Features

- **Forensic Cleaning**: Removes UI leaks, wire-service prefixes, metadata, and temporal markers to avoid shortcut learning.
- **Deep + Heuristic Stack**: BERT classifier + TF-IDF baseline + BERTopic clusters for interpretability.
- **Interactive Dashboard**: One-click Streamlit app with automatic model download on first run.
- **Reproducible Pipeline**: End-to-end scripts for dataset prep, training, evaluation, and topic visualization.
- **Portable Models**: Upload utility for pushing trained models to Hugging Face Hub.

---

## ⚡ Quick Start (Dashboard)

The easiest way to try TruthLens is via the interactive dashboard. On first run, it will auto-download any required models.

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch the Command Center
streamlit run app.py
```

- The app opens at **http://localhost:8501**.
- Paste a news article and view:
  - Fake probability
  - Attention heatmaps
  - External fact-checking (NLI-style) cues

---

## 🧭 Project Structure

```
TruthLens/
├── data/                      # Dataset storage (e.g., balanced_dataset_200k.parquet)
├── models/                    # Trained weights and checkpoints
├── plots/                     # Generated evaluation figures
├── src/
│   ├── config.py              # Paths & hyperparameters
│   ├── create_balanced_dataset.py  # Data balancing pipeline
│   ├── train_bert_v2.py       # MAIN training (Forensic BERT, V12 Strict)
│   ├── train_tfidf.py         # TF-IDF + Logistic Regression baseline
│   ├── train_topics.py        # BERTopic training
│   ├── evaluate_models.py     # Confusion matrices, ROC, calibration
│   ├── visualize_topics.py    # Topic charts/exports
│   └── upload.py              # Hugging Face Hub uploader
├── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Reproduction Pipeline

### 1) Data Preparation

A pre-processed dataset is included:

- `data/balanced_dataset_200k.parquet`

(Optional) Regenerate a balanced dataset from raw partitions:

```bash
python src/create_balanced_dataset.py
```

### 2) Model Training

**Forensic BERT (V2) — Primary Model**  
Trained with data without punctuation.

```bash
python src/train_bert_v1.py
# Output: models/bert_finetuned/
```

**Forensic BERT (V2) — Primary Model**  
Trained with **Strict** cleaning to remove artifacts.

```bash
python src/train_bert_v2.py
# Output: models/bert_final/
```

**Heuristic Baseline (TF-IDF)**  
Detects “lazy” fakes via keyword patterns.

```bash
python src/train_tfidf.py
# Output: models/tfidf_logreg_robust.joblib
```

**Topic Model (BERTopic)**  
Clusters articles into semantic themes.

```bash
python src/train_topics.py
# Output: models/bertopic_model/
```

### 3) Evaluation & Metrics

Generate confusion matrices, ROC curves, calibration plots, and summaries:

```bash
python src/evaluate_models.py
# Results in: plots/ (e.g., cm_new_bert.png, roc_*.png)
```

### 4) Topic Visualization

Export static visualizations for topic clusters:

```bash
python src/visualize_topics.py
# Results in: plots/topics_*.png
```

---

## 🧪 Forensic Cleaning (V12 Strict)

To prevent metadata leakage and date-based shortcuts, we remove:

- **UI Leaks**: “Click here”, “Join our newsletter”, “View Gallery”
- **Agency Headers**: Reuters, AP, AFP, CNN, etc.
- **Metadata**: “Photo credit”, “Editor”, “Reporting by”
- **Temporal Markers**: Month names (January, February, …)

Implementation details live in `train_bert_v2.py` and shared utilities referenced by `config.py`.

---

## ☁️ Model Deployment & Portability

Use `src/upload.py` to push trained models to the Hugging Face Hub. This lets the Streamlit app fetch them automatically on any machine.

```bash
# Prerequisite: export your HF write token securely
export HUGGINGFACE_HUB_TOKEN=hf_xxx

# Upload your final model(s)
python src/upload.py
```

**What it does:**

- Pushes `models/bert_final/` (and any other configured paths) to your specified repositories.

> 🔒 **Security Tip:** Do **not** hard-code tokens in `config.py`. Use environment variables or your CI secret store.  
> ✅ Pin the owner in `repo_id`, e.g. `"YourUser/fake-news-bert-v2"`, so uploads never go to the wrong account.

---

## 🚀 CLI Recipes

**Train + Evaluate end-to-end**

```bash
python src/train_bert_v2.py && python src/train_bert_v2.py  && python src/train_tfidf.py  && python src/train_topics.py  && python src/evaluate_models.py
```

**Run Dashboard after training**

```bash
streamlit run app.py
```

**Upload trained models**

```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
python src/upload.py
```

---

## 🧩 Dependencies

Install from the pinned requirements:

```bash
pip install -r requirements.txt
```

Typical stack (subset): `transformers`, `torch`, `huggingface_hub`, `scikit-learn`, `pandas`, `numpy`, `bertopic`, `umap-learn`, `matplotlib`, `streamlit`.

---

## 🧰 Troubleshooting

- **403/401 during upload**  
  Ensure `HUGGINGFACE_HUB_TOKEN` has write scope and that `repo_id` is in your namespace (e.g., `YourUser/...`) or you’re a collaborator.

- **Models not found in the app (first run)**  
  Check internet connectivity and that model repo names match `config.py`. The app downloads on demand.

- **CUDA Out of Memory**  
  Lower `BATCH_SIZE`, reduce `MAX_LEN`, or run on CPU by setting `CUDA_VISIBLE_DEVICES=""`.

- **Topic modeling errors**  
  Ensure `umap-learn` and `hdbscan` are installed (if you use HDBSCAN-based clustering).

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- Hugging Face Transformers & Hub
- Streamlit
- BERTopic

---

## 🧑🏻‍💻 Author

Simone De Giorgi - [GitHub](https://github.com/simo-dg) | [LinkedIn](https://www.linkedin.com/in/simone-de-giorgi/)
