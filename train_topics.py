from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import re
from tqdm import tqdm

# --- CONFIG ---
BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"

# --- REUSE YOUR CLEANING LOGIC ---
def clean_text_v12(text):
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

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA / "balanced_dataset_200k.parquet")
    
    # Sample for speed? (Optional: 200k takes ~20 mins on GPU)
    # df = df.sample(50000, random_state=42) 
    
    print("Cleaning data...")
    docs = [clean_text_v12(t) for t in tqdm(df['text'], desc="Cleaning")]
    # Remove short docs to prevent noise
    docs = [d for d in docs if len(d) > 50]

    print("Initializing BERTopic...")
    # 'all-MiniLM-L6-v2' is fast and effective for English
    # n_gram_range=(1, 2) captures "white house" or "tax cuts"
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        vectorizer_model=vectorizer_model,
        verbose=True,
        min_topic_size=50 # Avoid tiny micro-topics
    )

    print("Fitting model (This takes time)...")
    topic_model.fit_transform(docs)

    print("Saving model...")
    # Save the model with safetensors for efficiency
    topic_model.save(MODELS / "bertopic_model", serialization="safetensors")
    # Save topic info and c_tf_idf_ as CSV and NPY
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(MODELS / "bertopic_topic_info.csv", index=False)
    if getattr(topic_model, "c_tf_idf_", None) is not None:
        import numpy as np
        np.save(MODELS / "bertopic_c_tf_idf.npy", topic_model.c_tf_idf_)
    
    # Print info
    freq = topic_model.get_topic_info()
    print(freq.head(10))
    
    # Save visualization (Optional)
    fig = topic_model.visualize_barchart(top_n_topics=10)
    fig.write_html(BASE / "plots" / "topics.html")

if __name__ == "__main__":
    main()