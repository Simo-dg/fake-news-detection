import config
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from pathlib import Path

# --- CONFIGURATION ---
api = HfApi()
MODELS_DIR = config.MODELS_DIR

# Define your Repo IDs
REPO_BERT_V1 = config.HF_MODEL_ID
REPO_BERT_V2 = config.HF_MODEL_ID_V2
REPO_BERTOPIC = config.REPO_BERTOPIC
REPO_TFIDF = config.REPO_TFIDF


print(f"Starting upload process from: {MODELS_DIR}")

# 1. UPLOAD BERT V1 (Folder)
print(f"\nProcessing BERT V1 -> {REPO_BERT_V1}...")
try:
    create_repo(REPO_BERT_V1, private=False, exist_ok=True)
    upload_folder(
        folder_path=MODELS_DIR / "bert_finetuned_cleaned",
        repo_id=REPO_BERT_V1,
        repo_type="model"
    )
    print("Successfully uploaded BERT V1.")
except Exception as e:
    print(f"Failed to upload BERT V1: {e}")

# 2. UPLOAD BERT V2 (Folder)
print(f"\nProcessing BERT V2 -> {REPO_BERT_V2}...")
try:
    create_repo(REPO_BERT_V2, private=False, exist_ok=True)
    upload_folder(
        folder_path=MODELS_DIR / "bert_final",
        repo_id=REPO_BERT_V2,
        repo_type="model"
    )
    print("Successfully uploaded BERT V2.")
except Exception as e:
    print(f"Failed to upload BERT V2: {e}")

# 3. UPLOAD BERTopic (Folder)
print(f"\nProcessing BERTopic -> {REPO_BERTOPIC}...")
try:
    create_repo(REPO_BERTOPIC, private=False, exist_ok=True)
    
    bertopic_path = MODELS_DIR / "bertopic_model"
    if bertopic_path.exists():
        upload_folder(
            folder_path=bertopic_path,
            repo_id=REPO_BERTOPIC,
            repo_type="model"
        )
        print("Successfully uploaded BERTopic.")
    else:
        print(f"BERTopic folder not found at {bertopic_path}. Skipping.")
except Exception as e:
    print(f"Failed to upload BERTopic: {e}")

# 4. UPLOAD TF-IDF (Single File)
print(f"\nProcessing TF-IDF -> {REPO_TFIDF}...")
try:
    create_repo(REPO_TFIDF, private=False, exist_ok=True)
    
    tfidf_path = MODELS_DIR / "tfidf_logreg_robust.joblib"
    if tfidf_path.exists():
        upload_file(
            path_or_fileobj=tfidf_path,
            path_in_repo="tfidf_logreg_robust.joblib",
            repo_id=REPO_TFIDF,
            repo_type="model"
        )
        print("Successfully uploaded TF-IDF model.")
    else:
        print(f"TF-IDF file not found at {tfidf_path}. Skipping.")
except Exception as e:
    print(f"Failed to upload TF-IDF model: {e}")

print("\nUpload process completed.")