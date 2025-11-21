from pathlib import Path
import numpy as np
import pandas as pd
import torch
import re
import gc
import glob
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
CHUNK_OVERLAP = 128
BATCH_SIZE = 16

# GPU Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """
    Removes data artifacts that cause the model to 'cheat'.
    Removes signatures like 'WASHINGTON (Reuters) - '
    """
    if not isinstance(text, str): return ""
    
    # Remove "WASHINGTON (Reuters) -" patterns
    text = re.sub(r'^[A-Z\s\.,]+ \((Reuters|AP|AFP|CNN)\)\s*-\s*', '', text)
    # Remove generic location headers "LONDON -"
    text = re.sub(r'^[A-Z\s]{3,}\s+-\s*', '', text)
    # Remove Twitter handles often found in fake news
    text = re.sub(r'@\w+', '', text)
    
    return text.strip()

def compute_metrics_chunk(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy_chunk": acc, "f1_macro_chunk": f1}

def make_doclevel_metric(eval_doc_ids):
    eval_doc_ids = np.array(eval_doc_ids)
    
    def _compute(p):
        logits = p.predictions
        labels = p.label_ids
        
        # Get unique document IDs
        unique_docs = np.unique(eval_doc_ids)
        doc_logits_list = []
        doc_labels_list = []
        
        for did in unique_docs:
            # Find all chunks belonging to this document
            idx = np.where(eval_doc_ids == did)[0]
            
            # Get logits for these chunks: Shape (num_chunks, 2)
            chunk_logits = logits[idx]
            
            # --- SMART AGGREGATION LOGIC ---
            
            # OPTION A: Pure Max-Pooling (Most Aggressive)
            # We look at the 'Fake' class (index 1). 
            # We take the chunk that has the HIGHEST score for 'Fake'.
            fake_scores = chunk_logits[:, 1] 
            best_chunk_idx = np.argmax(fake_scores)
            
            # We use the logits of that specific "most suspicious" chunk
            doc_logits_list.append(chunk_logits[best_chunk_idx])
            
            # --- OPTION B (Alternative): Top-3 Average (More Stable) ---
            # If you find Option A is too noisy (too many false alarms),
            # uncomment the lines below to average the top 3 most suspicious chunks.
            # 
            # sorted_indices = np.argsort(chunk_logits[:, 1])[::-1] # Sort desc by Fake score
            # top_k = min(3, len(chunk_logits))
            # top_logits = chunk_logits[sorted_indices[:top_k]]
            # doc_logits_list.append(top_logits.mean(axis=0))
            
            # -------------------------------

            doc_labels_list.append(labels[idx][0])
            
        doc_logits = np.vstack(doc_logits_list)
        y_true = np.array(doc_labels_list)
        y_pred = doc_logits.argmax(-1)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec}
        
    return _compute

def load_data(data_dir):
    """Loads data prioritizing the balanced parquet file."""
    balanced_file = data_dir / "balanced_dataset_200k.parquet"
    
    if balanced_file.exists():
        print(f"Loading balanced dataset: {balanced_file.name}")
        df = pd.read_parquet(balanced_file)
    else:
        print("Loading raw parquet parts...")
        parquet_files = sorted(glob.glob(str(data_dir / "clean_df_part_*.parquet")))
        if not parquet_files:
            # Fallback to CSV if no parquet
            raise FileNotFoundError("Nessun file parquet o CSV trovato per il caricamento dei dati.")
        # Load just first file for testing, or all for full training
        dfs = [pd.read_parquet(f) for f in parquet_files[:2]] # Limit to 2 files if debugging
        df = pd.concat(dfs, ignore_index=True)

    if 'y_fake' in df.columns:
        df = df.rename(columns={'y_fake': 'label'})
        
    return df

# --- CUSTOM TRAINER ---

class DocLevelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Safe extraction of metadata
        labels = inputs.get("labels")
        
        # Remove metadata columns that the model doesn't expect
        # This prevents "forward() got an unexpected keyword argument"
        model_inputs = {k: v for k, v in inputs.items() if k not in ["doc_id", "labels"]}
        
        outputs = model(**model_inputs)
        logits = outputs.logits
        
        # Handle weights ensuring device compatibility
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None
            
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- MAIN ---

def main():
    # 1. LOAD DATA
    df = load_data(DATA)
    df["label"] = df["label"].astype(int)
    
    print(f"Total loaded: {len(df):,} articles")
    
    # 2. CLEANING ARTIFACTS (Crucial step for real-world performance)
    print("Cleaning text artifacts (Reuters, AP headers)...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove duplicates after cleaning
    df = df.drop_duplicates(subset=['text'], keep='first')
    df = df[df['text'].str.len() > 100] # Remove tiny articles
    print(f"Remaining after cleaning: {len(df):,}")
    
    # Assign stable Doc IDs
    df["doc_id"] = np.arange(len(df))
    
    # 3. SPLIT
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    
    # Calculate Weights
    class_counts = train_df['label'].value_counts().sort_index()
    weights = len(train_df) / (2 * class_counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float32)
    print(f"Class Weights: {class_weights}")

    # 4. TOKENIZATION & CHUNKING (Optimized with .map)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def preprocess_and_chunk(examples):
        tokenized = tok(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=CHUNK_OVERLAP,
            return_overflowing_tokens=True,
            padding=False
        )
        
        # Map chunks back to original doc_id and label
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        
        # "examples" is a dictionary of lists. We use sample_map to duplicate the metadata
        tokenized["label"] = [examples["label"][i] for i in sample_map]
        tokenized["doc_id"] = [examples["doc_id"][i] for i in sample_map]
        
        return tokenized

    print("Tokenizing...")
    # Convert to HF Dataset for speed
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    
    train_tok = train_ds.map(
        preprocess_and_chunk, 
        batched=True, 
        remove_columns=train_ds.column_names,
        desc="Chunking Train"
    )
    test_tok = test_ds.map(
        preprocess_and_chunk, 
        batched=True, 
        remove_columns=test_ds.column_names,
        desc="Chunking Test"
    )
    
    print(f"Train Chunks: {len(train_tok):,}")
    print(f"Test Chunks: {len(test_tok):,}")

    # 5. TRAINING SETUP
    coll = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    
    # We need to save the doc_ids for the metric function before the trainer hides them
    test_doc_ids = test_tok['doc_id']
    doc_metric_fn = make_doclevel_metric(test_doc_ids)
    
    def compute_metrics_combined(p):
        out = compute_metrics_chunk(p)
        out.update(doc_metric_fn(p))
        return out

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=str(MODELS/"bert_finetuned_cleaned"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
        remove_unused_columns=False, # CRITICAL: Keeps doc_id in the inputs
        report_to="none"
    )

    trainer = DocLevelTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        processing_class=tok,
        data_collator=coll,
        compute_metrics=compute_metrics_combined,
        class_weights=class_weights
    )

    # 6. RUN
    print("Starting training...")
    trainer.train()
    
    # Save
    print("Saving model...")
    trainer.save_model(MODELS/"bert_finetuned_cleaned")
    tok.save_pretrained(MODELS/"bert_finetuned_cleaned")
    
    # Final Eval
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()