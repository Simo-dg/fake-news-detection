from huggingface_hub import HfApi, create_repo, upload_folder
import config

api = HfApi()
repo_id = config.HF_MODEL_ID
create_repo(repo_id, private=False, exist_ok=True)

upload_folder(
    folder_path=config.MODELS_DIR / "bert_finetuned",
    repo_id=repo_id,
    repo_type="model"
)

repo_id2 = config.HF_MODEL_ID_V2
create_repo(repo_id2, private=False, exist_ok=True)

upload_folder(
    folder_path=config.MODELS_DIR / "bert_final",
    repo_id=repo_id2,
    repo_type="model"
)


print("✅ Uploaded! View at: https://huggingface.co/Simingasa/fake-news-bert-finetuned")
print("✅ Uploaded! View at: https://huggingface.co/Simingasa/fake-news-bert-v2")