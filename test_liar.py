import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# --- CONFIGURATION ---
# YOUR MODEL ID ON HUGGING FACE
MODEL_ID = "Simingasa/fake-news-bert-finetuned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def map_liar_labels(row):
    """
    Maps LIAR's 6-point scale to Binary (0: Real, 1: Fake).
    
    LIAR Labels:
    0: false        -> FAKE (1)
    1: half-true    -> REAL (0) - giving benefit of doubt
    2: mostly-true  -> REAL (0)
    3: pants-fire   -> FAKE (1)
    4: true         -> REAL (0)
    5: barely-true  -> FAKE (1)
    """
    label = row['label']
    if label in [0, 3, 5]:
        return 1 # FAKE
    else:
        return 0 # REAL

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. LOAD LIAR DATASET
    print("\n⬇️  Downloading LIAR dataset from Hugging Face...")
    dataset = load_dataset("liar")
    
    # We only need the 'test' split
    df_test = pd.DataFrame(dataset['test'])
    print(f"Original Test Size: {len(df_test)}")
    
    # 2. PREPROCESS LABELS
    print("🔄 Mapping labels (6-class -> Binary)...")
    df_test['binary_label'] = df_test.apply(map_liar_labels, axis=1)
    
    print("Class Distribution:")
    print(df_test['binary_label'].value_counts().rename({0: 'REAL', 1: 'FAKE'}))
    
    # 3. LOAD YOUR MODEL FROM HUB
    print(f"\n🤖 Downloading your model: {MODEL_ID}...")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. RUN PREDICTION LOOP
    print("\n🚀 Running Inference on LIAR statements...")
    predictions = []
    texts = df_test['statement'].tolist()
    
    # We use a simple loop here (no chunking) because LIAR statements are short
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        inputs = tok(
            batch_texts, 
            truncation=True, 
            max_length=512, 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            
    # 5. EVALUATION
    y_true = df_test['binary_label'].tolist()
    y_pred = predictions
    
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS: {MODEL_ID} vs LIAR")
    print("="*60)
    print(f"ACCURACY: {acc:.2%}")
    print("-" * 60)
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
    
    # 6. ERROR ANALYSIS
    df_test['prediction'] = y_pred
    
    # Filter for False Negatives (Actual: FAKE, Predicted: REAL)
    errors = df_test[(df_test['binary_label'] == 1) & (df_test['prediction'] == 0)]
    
    print("\n" + "="*60)
    print("FAILURE ANALYSIS: Lies that fooled the model")
    print("="*60)
    
    for idx, row in errors.head(5).iterrows():
        print(f"Statement: \"{row['statement']}\"")
        print(f"Speaker:   {row['speaker']}")
        print(f"Label:     {row['label']} (Fake/Barely True/Pants Fire)")
        print("-" * 40)

    print(f"\nSummary: The model missed {len(errors)} fake statements out of {len(df_test[df_test['binary_label']==1])} total fakes.")

if __name__ == "__main__":
    main()