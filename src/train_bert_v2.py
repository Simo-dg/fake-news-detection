from pathlib import Path
import numpy as np
import pandas as pd
import torch
import re
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import config

# --- CONFIGURATION ---
DATA = config.DATA_DIR
MODELS = config.MODELS_DIR

MODEL_NAME = config.MODEL_NAME
MAX_LENGTH = config.MAX_LENGTH
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
DEVICE = config.DEVICE

# Disable parallelism to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = DEVICE
print(f"Using device: {device}")

# --- CLEANING FUNCTION (V12: BERT OPTIMIZED) ---
def clean_text_bert(text):
    """
    Removes specific artifacts found during TF-IDF analysis but keeps
    punctuation so BERT can understand sentence structure.
    """
    if not isinstance(text, str): return ""
    
    text = text.lower()
    
    # 1. THE KILL LIST (Artifacts from TF-IDF V11)
    # We remove these specific phrases/words that were leaking labels.
    # Note: We use ' ' replacement to avoid merging words.
    patterns = [
        # Real News UI Leaks (-29 coefficient)
        r'advertisement', 
        r'reading main', r'main story', r'continue reading',
        r'president elect', r'source text', r'rights reserved', 
        r'copyright', r'misstated', r'company coverage',
        r'newsletter',
        
        # Fake News UI Leaks (+33 coefficient)
        r'fact box', r'story fact', r'fact check', 
        r'add cents', r'add your two cents', 
        r'readers think', r'view gallery', r'featured image',
        r'read more', r'click here', r'sign up', 
        r'proactiveinvestors', 
        r'visit our', r'check out'
    ]
    
    combined_pattern = re.compile('|'.join(patterns))
    text = combined_pattern.sub(' ', text)

    # 2. EDITORIAL / CREDITS
    meta_words = [
        r'\bphoto\b', r'\bimage\b', r'\bcredit\b', 
        r'\beditor\b', r'\bediting\b', 
        r'\bwriter\b', r'\bphotograph\b', 
        r'\bcaption\b', r'\breporting\b'
    ]
    for p in meta_words:
        text = re.sub(p, ' ', text)

    # 3. AGENCIES & DOMAINS
    agencies = [
        r'\b(reuters|ap|afp|upi|bloomberg|cnbc|cnn|bbc|nyt|new york times|washington post)\b',
        r'\b(nytimes|breitbart|christian post|consortiumnews|daily caller)\b',
        r'\b(calif|gmt|est|pst)\b'
    ]
    for pattern in agencies:
        text = re.sub(pattern, ' ', text)

    # 4. TEMPORAL (Months) - Prevents model guessing based on date
    months = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b'
    text = re.sub(months, ' ', text)

    # 5. STANDARD CLEANUP
    text = re.sub(r'@\w+', '', text)       # Twitter handles
    text = re.sub(r'http\S+', '', text)    # URLs
    
    # CRITICAL DIFFERENCE FOR BERT:
    # We DO NOT remove punctuation [^\w\s]. 
    # BERT needs periods and commas to understand sentence flow.
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- METRICS ---
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec}

# --- MAIN ---
def main():
    # 1. LOAD
    print("Loading Data...")
    # Ensure this path matches your actual parquet location
    df = pd.read_parquet(DATA / "balanced_dataset_200k.parquet")
    
    # 2. CLEAN (Apply V12)
    print("Cleaning Text (BERT Optimized - Keeping Punctuation)...")
    df['text'] = df['text'].apply(clean_text_bert)
    
    # Filter tiny rows (BERT needs some context)
    df = df[df['text'].str.len() > 100]
    
    # Ensure labels are integers
    if "label" not in df.columns and "y_fake" in df.columns:
        df = df.rename(columns={"y_fake": "label"})
    df["label"] = df["label"].astype(int)
    
    print(f"Remaining articles: {len(df):,}")
    
    # 3. SPLIT
    print("Splitting...")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    
    # 4. TOKENIZATION
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok_func(examples):
        return tok(examples["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    print("Tokenizing...")
    # Remove raw text columns to save RAM, keep input_ids/attention_mask
    train_ds = train_ds.map(tok_func, batched=True)
    test_ds = test_ds.map(tok_func, batched=True)
    
    cols_to_remove = [c for c in train_ds.column_names if c not in ["input_ids", "attention_mask", "label"]]
    train_ds = train_ds.remove_columns(cols_to_remove)
    test_ds = test_ds.remove_columns(cols_to_remove)

    # 5. TRAINER SETUP
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Use DataCollatorWithPadding for dynamic padding (faster than static padding)
    data_collator = DataCollatorWithPadding(tokenizer=tok)
    
    args = TrainingArguments(
        output_dir=str(MODELS / "bert_final"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,             # Low learning rate for stability
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available()  # Mixed Precision
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. RUN
    print("Starting BERT Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model(MODELS / "bert_final")
    tok.save_pretrained(MODELS / "bert_final")
    
    print("Final Evaluation on Test Set:")
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()