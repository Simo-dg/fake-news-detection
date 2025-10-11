# 📰 Fake News Detection

A complete **Fake News Detection** system using both classical NLP (TF-IDF + Logistic Regression) and modern Transformer models (**DistilBERT**).  
Includes training scripts, evaluation plots, and an interactive **Streamlit dashboard** for testing real and fake news articles.

---

## 🚀 Overview

- **TF-IDF + Logistic Regression** baseline for fast, interpretable results.  
- **DistilBERT** used as frozen encoder and fine-tuned end-to-end.  
- **Evaluation** with confusion matrices, ROC/AUC curves, and metrics.  
- **Interactive Streamlit app** for article classification.  
- **Fine-tuned model hosted on Hugging Face** for instant download.

---

## 📦 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Simo-dg/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit dashboard
```bash
streamlit run app.py
```

> The app automatically downloads the fine-tuned model from Hugging Face.

---

## 🤗 Model Download

Fine-tuned model available here:  
🔗 [https://huggingface.co/Simingasa/fake-news-bert-finetuned](https://huggingface.co/Simingasa/fake-news-bert-finetuned)

Load it directly in Python:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "Simingasa/fake-news-bert-finetuned"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForSequenceClassification.from_pretrained(model_id)

```

---

## 📁 Project Structure
```
fake-news-detection/
│
├── data/                         # Dataset (ignored in Git)
│   ├── True.csv
│   └── Fake.csv
│
├── models/                       # Trained models (ignored in Git)
│   └── bert_finetuned/
│
├── plots/                        # Evaluation plots
│
├── utils_data.py                 # Helper for dataset loading
├── train_tfidf.py                # Train TF-IDF + Logistic Regression
├── train_bert_feature_extractor.py  # Train BERT as frozen encoder
├── finetune_bert.py              # Fine-tune DistilBERT end-to-end
├── evaluate_models.py            # Compare models and generate plots
├── app.py                        # Streamlit dashboard
├── upload_to_huggingface.py      # Upload model to Hugging Face
│
├── requirements.txt
└── README.md
```


---

## 🧠 Example Predictions

**✅ Real Article**
> NASA announced a new launch window for its Artemis I mission, marking the first step in returning humans to the Moon.

**❌ Fake Article**
> NASA scientists confirmed the discovery of ancient alien structures on the Moon, according to leaked Artemis I photos.

---

## ☁️ Deployments

### ▶️ Local
```bash
streamlit run app.py
```



---

## 🧩 Dataset

Dataset: [Fake News Detection Datasets (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)  
It contains 21,417 real and 23,481 fake news articles collected from verified sources.

---

## 📚 Technologies Used

| Category | Libraries |
|-----------|------------|
| Data & ML | pandas, scikit-learn, numpy |
| Deep Learning | PyTorch, Transformers, Datasets |
| Visualization | matplotlib, seaborn |
| App | Streamlit |
| Hosting | Hugging Face Hub |

---

## 👨‍💻 Author

**Simone De Giorgi**  
📍 MSc — Economics (ML focus)
💼 GitHub → [https://github.com/Simo-dg](https://github.com/Simo-dg)  
🤗 Hugging Face → [https://huggingface.co/Simingasa](https://huggingface.co/Simingasa)

---

## 📝 License
Released under the **MIT License**.  
You are free to use, modify, and share this project for educational or research purposes.
