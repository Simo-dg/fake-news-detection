from huggingface_hub import HfApi, create_repo, upload_folder

api = HfApi()
repo_id = "Simingasa/fake-news-bert-finetuned"
create_repo(repo_id, private=False, exist_ok=True)

upload_folder(
    folder_path="models/bert_finetuned",
    repo_id=repo_id,
    repo_type="model"
)

repo_id2 = "Simingasa/fake-news-bert-v2"
create_repo(repo_id2, private=False, exist_ok=True)

upload_folder(
    folder_path="models/bert_final",
    repo_id=repo_id2,
    repo_type="model"
)


print("✅ Uploaded! View at: https://huggingface.co/Simingasa/fake-news-bert-finetuned")
print("✅ Uploaded! View at: https://huggingface.co/Simingasa/fake-news-bert-v2")