from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils_data import load_true_fake
from sklearn.model_selection import train_test_split


BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)
PLOTS = BASE / "plots";  PLOTS.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec}

def main():
    df = load_true_fake(DATA/"True.csv", DATA/"Fake.csv")
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tok_fn(ex): return tok(ex["text"], truncation=True)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    coll = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODELS/"bert_finetuned"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tok, data_collator=coll,
        compute_metrics=compute_metrics
    )
    trainer.train()

    trainer.save_model(MODELS/"bert_finetuned")
    tok.save_pretrained(MODELS/"bert_finetuned")

    final_metrics = trainer.evaluate()
    with open(BASE/"evaluation_bert_finetuned.txt", "w") as f:
        for k,v in final_metrics.items():
            f.write(f"{k}: {v}\n")
    print("Saved fine-tuned model to:", MODELS/"bert_finetuned")

if __name__ == "__main__":
    main()
