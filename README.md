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

An example of a true article you can test: (from BBC News)

> The first official data in weeks on the US job market is out, and it showed a surprising pick-up in hiring after a lacklustre summer.
> Employers added 119,000 jobs in September, more than double what many analysts had expected, but the unemployment rate ticked up from 4.3% to 4.4%, the Labor Department figures showed.
> The US government shutdown, which ended last week after more than a month, had delayed publication of the figures for nearly seven weeks, leaving policymakers guessing about the state of the job market at a delicate moment.
> Job growth has barely budged since April, raising pressure on the central bank to cut interest rates to support the economy.
> But policymakers at the US central bank, the Federal Reserve, have been divided about the need for further interest rate cuts. In addition to the health of the job market, they are also monitoring price inflation that ticked up to 3% in September, above the 2% rate the bank wants to see.
> Looming over the debate are questions like whether artificial intelligence (AI) will dampen demand for workers over the long term and how a crackdown on immigration is changing labour supply and demand.
> Businesses are also wrestling with cutbacks to government spending, new tariff costs and uncertain consumer demand.
> A private report this month by outplacement firm Challenger, Gray & Christmas found the number of job cuts in October hit the highest number for the month since 2003, as high-profile companies including Amazon, Target and UPS announced reductions.
> On Thursday, telecoms giant Verizon also said it was cutting more than 13,000 jobs, citing in part "changes in technology and in the economy" for the move.
> The announcements have raised concerns about cracks in what has been seen as a "low-hire, low-fire" job market.
> But evidence of wider deterioration has been elusive, as claims for unemployment benefits remain stable.
> Health care firms, restaurants and bars led the job gains in September, while transportation and warehousing firms, manufacturers and the government shed jobs.
> "The September jobs report may be backward looking but offers reassurance that the labour market wasn't crumbling before the government shutdown," said Nancy Vanden Houten, lead economist at Oxford Economics.
> However, she noted noting that the data from October is likely to be weaker, due to government layoffs.
> Limited hiring has already prompted the ranks of people without work more than six months to swell this year, though their numbers dipped a bit in September.
> Unusually, the strains have been particularly pronounced among those with college degrees. The unemployment rate for that group rose to 2.8% in September, up from from 2.3% a year earlier.
> "It's been pretty challenging," said Mason Leposavic, who has applied to thousands of jobs since graduating in May 2024 from the Rochester Institute of Technology.

An example of a fake article you can test:(from a report of the European Commission)

> VATICAN CITY – News outlets around the world are reporting on the news that Pope Francis has made the unprecedented decision to endorse a US presidential candidate. His statement in support of Donald Trump was released from the Vatican this evening: “I have been hesitant to offer any kind of support for either candidate in the US presidential election but I now feel that to not voice my concern would be a dereliction of my duty as the Holy See. A strong and free America is vitally important in maintaining a strong and free world and in that sense what happens in American elections affects us all. The Rule of Law is the backbone of the American government as it is in any nation that strives for freedom and I now fear that the Rule of Law in America has been dealt a dangerous blow. The FBI, in refusing to recommend prosecution after admitting that the law had been broken on multiple occasions by Secretary Clinton, has exposed itself as corrupted by political forces that have become far too powerful. Though I don’t agree with Mr. Trump on some issues, I feel that voting against the powerful political forces that have corrupted the entire American federal government is the only option for a nation that desires a government that is truly for the people and by the people. For this primary reason I ask, not as the Holy Father, but as a concerned citizen of the world that Americans vote for Donald Trump for President of the United States.” Sources within the Vatican reportedly were aware that the Pope had been discussing the possibility of voicing his concern in the US presidential election but apparently were completely unaware that he had made a decision on going forward with voicing this concern until his statement was released this evening from the Vatican. Stay tuned to WTOE 5 News for more on this breaking news.

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
│   ├── train_bert_v1.py       # INITIAL training (Forensic BERT, V1 Punctuation Removal)
│   ├── train_bert_v2.py       # MAIN training (Forensic BERT, Strict cleaning)
│   ├── train_tfidf.py         # TF-IDF + Logistic Regression baseline
│   ├── train_topics.py        # BERTopic training
│   ├── evaluate_models.py     # Confusion matrices, ROC, calibration
│   ├── visualize_topics.py    # Topic charts/exports
│   └── upload.py              # Hugging Face Hub uploader
├── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

[![](https://mermaid.ink/img/pako:eNqNV1tvozgU_iuWR900WpIJUNKWrkbKtY02TStCu5qd7IMDhlglwBgzbSfKf19fCAktmQ4PwT7-zs3n8zHZQC_xMbThycmGxITZYNNgK7zGDRs0lijDDQ0owSOiBC0jnPGVDWgESczm5KfE6Vb6InBCNkZrEr0KaY_jIyFOKVkj-jpIooSKhU9BEAg59kM8RUsc9ZH3FNIkj_2GXa56UZ4xTPtPodK5CC4DdLiQUB8re0vfM73zBthutycniziIkmdvhSgDU2cRA_6cnIBWqwWmva8jB-g2GPbcHhjNriez0ciZzK7FqkJm-TKkKF2BIWJoil4xBd8WUOL_ACN3qmws4H8KLh6fUOwxksTA7e-lDnr-drqA_CVNgSl6wn8t6ecvp0a7ewuc5DlrLmDzwM5-VAYhHHL34nVPUhyRGFdci2eO1ilfCDluvsqDIMI80j6KUOxhcGp1Plud5julQYRRLBJLnuOsMABGcfjWAY79ugDnLKFYpDfGiOUUK0GRXqfzBNopot9zzA5SlKZqymHY4PZuOJqC-cPEHdVUwqWIMzMOy2oouOv0JjNRO6V3esdT5-E3f6M2-xGP5JoTLxWkcFeY7wvKMhIQTLOacszzFNMfJMO-CONgdqj2dqvrQ6iGUYRyy49iBHpV-Z4ML2lPpq9QNt90FPuI-mA2vX_ntqzyo851HLxOfmBwn8cey5GIplahjymT-P7IccGjriq68_OeRhV6_CKl_vGU-vuU-jYYcxrFGfE-SMngOiVUijg_VLB_Y5we5tk8nqhRJmoo3Z3JI4keSW5wPLnBPrmBrBcjGSMeio7nJkB12dVquAHxA452x63JcAz-BNMkdHBYlm3n7oPKHTnl5eEw1OGY8IDCFas7GQ9xVjkblXmp9zYIN0mJVxRBDFXcA9Xfec7N-l50rJOYNnBG15O563ytaSODKMllbD3HnYx7A7cEV7zcjDebBbzJw1C0xDHiTfQmX6rIVCVFr0OhaDTb7a8DOuM1HzmP9RfMXGwPd8Ejunfuhg8Dd3I32-ErIT1kmJ7yZivevJ829yu9NBWNiFGM1hFhYl7RrLtSZJMXXidxgDnJeII1jV88U_ID7y4KMS5Pp9yNo8S8p9gnHiuZXyT6O7Uc3M1mI7kRcyUT12er9aW85OREelZice0opPqVgj3oUZdj1dbqEcYeYdQixDGSM3nY3kIkbw8yuM-zFWCJYpuSKu_iUpZO-EBaEm-hDFptwCkg9HqUkQB54qSANjd-Mz6wvOMLP-LC444YMgpR97gghBSUpZOzoiIKcjPeeYwi8A8uTqb0V8EVk739fSTu16ngqJx74uob4gD4SxCQKLI_YTMwAl_LGE2esP1Jt7qW1ymmrWfis5VtpC9Xb9RTmniFAf4BWGqbpllV1d-rriXNCucXgYUvS3UDn_um8aFzT_aGwruJrcAqLXSRvrxE7yzsBD7K-JcmRa82sIB1tdslaVmwV1NU8ZeHLks6a7JGWsHV4m1oJe-0so4aL4Gmzqm2q4zYsopZxTRN8UyTLNMUx-QWVbCcBjLrK6jBkBIf2ozmWINrTNdITOFGwBdQfvwvoM2HPg5QHrEFXMRbrpai-N8kWe80-VURrqAdoCjjszz1EcNDgnjb2UP4ccd0wL_0GbT1C92QRqC9gS_QblndTrt7YXUv9Y7VubywLA2-QvvsrH3BhR3TNMyz7rllWlsN_pR-9Xa3e252dLNrnhmm1T0zNci3hu_4rfpTI__bbP8H-jbCsA?type=png)](https://mermaid.live/edit#pako:eNqNV1tvozgU_iuWR900WpIJUNKWrkbKtY02TStCu5qd7IMDhlglwBgzbSfKf19fCAktmQ4PwT7-zs3n8zHZQC_xMbThycmGxITZYNNgK7zGDRs0lijDDQ0owSOiBC0jnPGVDWgESczm5KfE6Vb6InBCNkZrEr0KaY_jIyFOKVkj-jpIooSKhU9BEAg59kM8RUsc9ZH3FNIkj_2GXa56UZ4xTPtPodK5CC4DdLiQUB8re0vfM73zBthutycniziIkmdvhSgDU2cRA_6cnIBWqwWmva8jB-g2GPbcHhjNriez0ciZzK7FqkJm-TKkKF2BIWJoil4xBd8WUOL_ACN3qmws4H8KLh6fUOwxksTA7e-lDnr-drqA_CVNgSl6wn8t6ecvp0a7ewuc5DlrLmDzwM5-VAYhHHL34nVPUhyRGFdci2eO1ilfCDluvsqDIMI80j6KUOxhcGp1Plud5julQYRRLBJLnuOsMABGcfjWAY79ugDnLKFYpDfGiOUUK0GRXqfzBNopot9zzA5SlKZqymHY4PZuOJqC-cPEHdVUwqWIMzMOy2oouOv0JjNRO6V3esdT5-E3f6M2-xGP5JoTLxWkcFeY7wvKMhIQTLOacszzFNMfJMO-CONgdqj2dqvrQ6iGUYRyy49iBHpV-Z4ML2lPpq9QNt90FPuI-mA2vX_ntqzyo851HLxOfmBwn8cey5GIplahjymT-P7IccGjriq68_OeRhV6_CKl_vGU-vuU-jYYcxrFGfE-SMngOiVUijg_VLB_Y5we5tk8nqhRJmoo3Z3JI4keSW5wPLnBPrmBrBcjGSMeio7nJkB12dVquAHxA452x63JcAz-BNMkdHBYlm3n7oPKHTnl5eEw1OGY8IDCFas7GQ9xVjkblXmp9zYIN0mJVxRBDFXcA9Xfec7N-l50rJOYNnBG15O563ytaSODKMllbD3HnYx7A7cEV7zcjDebBbzJw1C0xDHiTfQmX6rIVCVFr0OhaDTb7a8DOuM1HzmP9RfMXGwPd8Ejunfuhg8Dd3I32-ErIT1kmJ7yZivevJ829yu9NBWNiFGM1hFhYl7RrLtSZJMXXidxgDnJeII1jV88U_ID7y4KMS5Pp9yNo8S8p9gnHiuZXyT6O7Uc3M1mI7kRcyUT12er9aW85OREelZice0opPqVgj3oUZdj1dbqEcYeYdQixDGSM3nY3kIkbw8yuM-zFWCJYpuSKu_iUpZO-EBaEm-hDFptwCkg9HqUkQB54qSANjd-Mz6wvOMLP-LC444YMgpR97gghBSUpZOzoiIKcjPeeYwi8A8uTqb0V8EVk739fSTu16ngqJx74uob4gD4SxCQKLI_YTMwAl_LGE2esP1Jt7qW1ymmrWfis5VtpC9Xb9RTmniFAf4BWGqbpllV1d-rriXNCucXgYUvS3UDn_um8aFzT_aGwruJrcAqLXSRvrxE7yzsBD7K-JcmRa82sIB1tdslaVmwV1NU8ZeHLks6a7JGWsHV4m1oJe-0so4aL4Gmzqm2q4zYsopZxTRN8UyTLNMUx-QWVbCcBjLrK6jBkBIf2ozmWINrTNdITOFGwBdQfvwvoM2HPg5QHrEFXMRbrpai-N8kWe80-VURrqAdoCjjszz1EcNDgnjb2UP4ccd0wL_0GbT1C92QRqC9gS_QblndTrt7YXUv9Y7VubywLA2-QvvsrH3BhR3TNMyz7rllWlsN_pR-9Xa3e252dLNrnhmm1T0zNci3hu_4rfpTI__bbP8H-jbCsA)

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

**Train BERT V1**  
Trained with data without punctuation.

```bash
python src/train_bert_v1.py
# Output: models/bert_finetuned/
```

**Train Bert V2**  
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

## 🧪 Cleaning

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
python src/train_bert_v1.py && python src/train_bert_v2.py  && python src/train_tfidf.py  && python src/train_topics.py  && python src/evaluate_models.py
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

## 🙌 Acknowledgments

- Hugging Face Transformers & Hub
- Streamlit
- BERTopic

---

## 🧑🏻‍💻 Author

Simone De Giorgi - [GitHub](https://github.com/simo-dg) | [LinkedIn](https://www.linkedin.com/in/simone-de-giorgi/)
