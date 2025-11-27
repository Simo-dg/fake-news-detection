import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config  # Importing the provided config file
from bertopic import BERTopic

# --- CONFIGURATION ---
# Using paths defined in config.py
DATA_PATH = config.DATA_DIR / "balanced_dataset_200k.parquet"
BERT_PATH = config.BERT_OLD_PATH
BERT_V2_PATH = config.BERT_FINAL_PATH
TFIDF_PATH = config.TFIDF_PATH
PLOTS_DIR = config.PLOTS_DIR
BERTOPIC_PATH = config.MODELS_DIR / "bertopic_model"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. LOAD DATA ---
print("Loading data...")
if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
else:
    df = pd.read_parquet(DATA_PATH)

    # Create a Test Split (Same random_state as training to ensure consistency)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_test = test_df['text'].tolist()
    y_test = test_df['label'].values

    print(f"Test data loaded: {len(test_df)} articles")

    # --- 2. SETUP DEVICE ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not found. Using CPU (Slow)...")

    # --- 3. LOAD TF-IDF ---
    print("\nLoading TF-IDF model...")
    if TFIDF_PATH.exists():
        tfidf_pipeline = joblib.load(TFIDF_PATH)

        t0 = time.time()
        y_pred_tfidf = tfidf_pipeline.predict(X_test)
        tfidf_latency = (time.time() - t0) / len(X_test) * 1000
        print(f"TF-IDF Complete. Latency: {tfidf_latency:.2f} ms/article")
        
        # Feature Importance Plot
        vectorizer = tfidf_pipeline.named_steps['tfidf']
        classifier = tfidf_pipeline.named_steps['clf']

        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]

        # Create DataFrame
        coef_df = pd.DataFrame({'word': feature_names, 'coef': coefs})
        top_20 = coef_df.reindex(coef_df.coef.abs().sort_values(ascending=False).index).head(20)

        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['red' if x > 0 else 'blue' for x in top_20['coef']]
        sns.barplot(data=top_20, x='coef', y='word', palette=colors)
        plt.title("Top 20 Most Predictive Words (TF-IDF)\nRed = FAKE indicator, Blue = REAL indicator")
        plt.xlabel("Coefficient Magnitude")
        
        # Save Plot
        out_path = PLOTS_DIR / "tfidf_top_features.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved TF-IDF plot to {out_path}")
        plt.close()
    else:
        print(f"Warning: TF-IDF model not found at {TFIDF_PATH}")
        y_pred_tfidf = np.zeros(len(y_test)) # Dummy for code continuity

    # --- 4. LOAD BERT V1 ---
    print("\nLoading BERT model (V1)...")
    if BERT_PATH.exists():
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
        model.to(device)
        model.eval()

        def predict_bert(texts, batch_size=32):
            preds = []
            timings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                t0 = time.time()

                # Tokenize
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)

                # Move Inputs to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Move Results back to CPU for NumPy
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

                timings.append((time.time() - t0) / len(batch))
                preds.extend(batch_preds)

            avg_latency = np.mean(timings) * 1000
            return np.array(preds), avg_latency

        print("Running BERT V1 predictions...")
        y_pred_bert, bert_latency = predict_bert(X_test)
        print(f" BERT V1 Complete. Latency: {bert_latency:.2f} ms/article")
    else:
        print(f"Warning: BERT V1 model not found at {BERT_PATH}")
        y_pred_bert = np.zeros(len(y_test))

    # --- 5. LOAD BERT V2 ---
    print("\nLoading BERT model (V2)...")
    if BERT_V2_PATH.exists():
        tokenizer_v2 = AutoTokenizer.from_pretrained(BERT_V2_PATH)
        model_v2 = AutoModelForSequenceClassification.from_pretrained(BERT_V2_PATH)
        model_v2.to(device)
        model_v2.eval()

        def predict_bert_v2(texts, batch_size=32):
            preds = []
            timings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                t0 = time.time()

                inputs = tokenizer_v2(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model_v2(**inputs)
                    logits = outputs.logits
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

                timings.append((time.time() - t0) / len(batch))
                preds.extend(batch_preds)

            avg_latency = np.mean(timings) * 1000
            return np.array(preds), avg_latency

        print("Running BERT V2 predictions...")
        y_pred_bert_v2, bert_v2_latency = predict_bert_v2(X_test)
        print(f" BERT V2 Complete. Latency: {bert_v2_latency:.2f} ms/article")
    else:
         print(f"Warning: BERT V2 model not found at {BERT_V2_PATH}")
         y_pred_bert_v2 = np.zeros(len(y_test))

    # --- 6. VISUALIZATION (Attention) ---
    text = "Breaking: The president signed a secret order to ban all cats."

    def plot_attention(model_path, text, title, filename):
        if not model_path.exists():
            return
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions
        n_layers = len(all_attentions)
        n_rows, n_cols = 2, 3 
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"Visualizing {n_layers} layers for {title}...")
        for i, layer_attn in enumerate(all_attentions):
            avg_attn = layer_attn.mean(dim=1).squeeze().numpy()
            sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens,
                        cmap="viridis", ax=axes[i], cbar=False)
            axes[i].set_title(f"Layer {i+1}")
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.suptitle(f"{title} Attention Patterns (Layer 1 to {n_layers})", fontsize=16, y=1.02)
        
        # Save Plot
        out_path = PLOTS_DIR / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention plot to {out_path}")
        plt.close()

    plot_attention(BERT_PATH, text, "DistilBERT V1", "attention_v1.png")
    plot_attention(BERT_V2_PATH, text, "DistilBERT V2", "attention_v2.png")

    # --- 7. BOOTSTRAP METRICS ---
    print("Calculating bootstrap metrics...")
    n_boot = 1000
    acc_tfidf, acc_bert, acc_bert_v2 = [], [], []
    prec_tfidf, prec_bert, prec_bert_v2 = [], [], []
    rec_tfidf, rec_bert, rec_bert_v2 = [], [], []
    f1_tfidf, f1_bert, f1_bert_v2 = [], [], []

    for _ in range(n_boot):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        yt = y_test[idx]
        
        # TF-IDF
        yp_tfidf = y_pred_tfidf[idx]
        acc_tfidf.append(accuracy_score(yt, yp_tfidf))
        prec_tfidf.append(precision_score(yt, yp_tfidf, zero_division=0))
        rec_tfidf.append(recall_score(yt, yp_tfidf, zero_division=0))
        f1_tfidf.append(f1_score(yt, yp_tfidf, zero_division=0))

        # BERT V1
        yp_bert = y_pred_bert[idx]
        acc_bert.append(accuracy_score(yt, yp_bert))
        prec_bert.append(precision_score(yt, yp_bert, zero_division=0))
        rec_bert.append(recall_score(yt, yp_bert, zero_division=0))
        f1_bert.append(f1_score(yt, yp_bert, zero_division=0))

        # BERT V2
        yp_bert_v2 = y_pred_bert_v2[idx]
        acc_bert_v2.append(accuracy_score(yt, yp_bert_v2))
        prec_bert_v2.append(precision_score(yt, yp_bert_v2, zero_division=0))
        rec_bert_v2.append(recall_score(yt, yp_bert_v2, zero_division=0))
        f1_bert_v2.append(f1_score(yt, yp_bert_v2, zero_division=0))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    results = [acc_tfidf, prec_tfidf, rec_tfidf, f1_tfidf, 
               acc_bert, prec_bert, rec_bert, f1_bert, 
               acc_bert_v2, prec_bert_v2, rec_bert_v2, f1_bert_v2]
    model_labels = ['TF-IDF']*4 + ['BERT V1']*4 + ['BERT V2']*4
    metric_labels = metrics*3

    df_boot = pd.DataFrame({'Metric': metric_labels, 'Model': model_labels, 'Values': [np.mean(x) for x in results],
                       'Lower': [np.percentile(x, 2.5) for x in results],
                       'Upper': [np.percentile(x, 97.5) for x in results]})

    # --- Plot Bar Chart ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    colors = ['#00BFAE', '#FF5252', '#FFA000']
    for i, metric in enumerate(metrics):
        for j, model in enumerate(['TF-IDF', 'BERT V1', 'BERT V2']):
            idx = (df_boot['Metric'] == metric) & (df_boot['Model'] == model)
            if idx.any():
                mean = df_boot[idx]['Values'].values[0]
                low = df_boot[idx]['Lower'].values[0]
                up = df_boot[idx]['Upper'].values[0]
                axes[i].bar(model, mean, yerr=[[mean-low], [up-mean]], capsize=10, color=colors[j], alpha=0.9, edgecolor='#222', linewidth=1.2)
                axes[i].annotate(f"{mean:.3f}", (model, mean), textcoords="offset points", xytext=(0,8), ha='center', fontsize=12, fontweight='bold', color='#222')
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_ylim(0.5, 1.0)
        axes[i].axhline(0.5, color='#888', linestyle=':', linewidth=1)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_color('#888')
        axes[i].spines['bottom'].set_color('#888')
        axes[i].yaxis.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            axes[i].set_ylabel('Score', fontsize=13)
    plt.suptitle('Bootstrap Metrics Comparison (95% CI)', fontsize=17, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save Bar Chart
    out_path = PLOTS_DIR / "bootstrap_metrics_bar.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Bootstrap Bar Chart to {out_path}")
    plt.close()

    # --- Plot Boxplot ---
    boot_df_raw = pd.DataFrame({
        'Accuracy': acc_tfidf + acc_bert + acc_bert_v2,
        'Precision': prec_tfidf + prec_bert + prec_bert_v2,
        'Recall': rec_tfidf + rec_bert + rec_bert_v2,
        'F1': f1_tfidf + f1_bert + f1_bert_v2,
        'Model': ['TF-IDF']*n_boot + ['BERT V1']*n_boot + ['BERT V2']*n_boot
    })

    boot_df_long = boot_df_raw.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=boot_df_long, x='Metric', y='Score', hue='Model', palette=['#00BFAE', '#FF5252', '#FFA000'], fliersize=2, linewidth=1.2)
    plt.title('Bootstrap Distribution of Metrics (Boxplot)', fontsize=17, fontweight='bold')
    plt.ylim(0.93, 0.98) 
    plt.ylabel('Score', fontsize=13)
    plt.xlabel('Metric', fontsize=13)
    plt.legend(title='Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save Box Plot
    out_path = PLOTS_DIR / "bootstrap_metrics_box.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Bootstrap Box Plot to {out_path}")
    plt.close()


try:
    topic_model = BERTopic.load(BERTOPIC_PATH)
    topics, _ = topic_model.transform(X_test)

    # Create analysis dataframe
    res_df = pd.DataFrame({
        'topic': topics,
        'correct': y_pred_bert == y_test
    })

    # Calculate error rate per topic (Filter for larger topics)
    topic_perf = res_df.groupby('topic')['correct'].agg(['mean', 'count'])
    topic_perf = topic_perf[topic_perf['count'] > 10] # Only topics with >10 samples

    # Retrieve topic labels from BERTopic (if available)
    try:
        topic_labels = topic_model.get_topic_info()
        label_dict = dict(zip(topic_labels['Topic'], topic_labels['Name'] if 'Name' in topic_labels.columns else topic_labels['Topic']))
        topic_perf = topic_perf.copy()
        topic_perf['label'] = topic_perf.index.map(label_dict)
        x_labels = topic_perf['label']
    except Exception as e:
        x_labels = topic_perf.index

    # Order by error rate
    topic_perf['error_rate'] = 1 - topic_perf['mean']
    topic_perf = topic_perf.sort_values('error_rate', ascending=False)
    x_labels = topic_perf['label'] if 'label' in topic_perf else topic_perf.index

    # table
    display_cols = ['label' if 'label' in topic_perf else 'topic', 'error_rate', 'count']
    print("\nTop 10 hardest topics for BERT (by error rate):")
    print(topic_perf[display_cols].head(10))

except Exception as e:
    print(f"Skipping Cluster Plot: Could not load BERTopic model ({e})")