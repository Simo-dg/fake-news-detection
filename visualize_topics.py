import matplotlib
matplotlib.use("Agg") # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
BASE = Path(__file__).parent.resolve()
MODELS = BASE / "models"
PLOTS = BASE / "plots"; PLOTS.mkdir(exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
PALETTE = "viridis"

def plot_top_keywords(topic_model, top_n=8):
    """
    Creates a 2x4 grid of bar charts showing the top words for the top 8 topics.
    """
    print("Generating Topic Keywords Grid...")
    
    # Get top topics (excluding -1 outlier)
    topic_info = topic_model.get_topic_info()
    top_topics = topic_info[topic_info["Topic"] != -1].head(top_n)
    
    # Prepare subplot grid
    rows = 2
    cols = (top_n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(top_topics.iterrows()):
        topic_id = row['Topic']
        # Get words and scores for this topic
        words_scores = topic_model.get_topic(topic_id)
        words = [x[0] for x in words_scores][:5] # Top 5 words
        scores = [x[1] for x in words_scores][:5]
        
        # Plot
        sns.barplot(x=scores, y=words, ax=axes[i], palette=PALETTE, hue=words, legend=False)
        axes[i].set_title(f"Topic {topic_id}: {row['Name'].split('_')[1]}...", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("")
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"Top {top_n} Topics by Keyword Importance", fontsize=20, y=1.05)
    out_path = PLOTS / "topics_keywords_grid.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_topic_sizes(topic_model, top_n=20):
    """
    Horizontal bar chart of topic sizes (number of documents).
    """
    print("Generating Topic Size Distribution...")
    
    info = topic_model.get_topic_info()
    # Filter out outlier (-1) and take top N
    data = info[info['Topic'] != -1].head(top_n)
    
    # Clean names for plotting (remove "1_", "2_" prefix)
    data['CleanName'] = data['Name'].apply(lambda x: " ".join(x.split("_")[1:4]))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=data, x="Count", y="CleanName", palette="mako", hue="CleanName", legend=False)
    
    plt.title(f"Top {top_n} Topics by Volume", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Documents")
    plt.ylabel("Topic Keywords")
    plt.grid(axis='x', alpha=0.3)
    
    out_path = PLOTS / "topics_sizes.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_similarity_heatmap(topic_model, top_n=15):
    """
    Heatmap showing which topics are semantically similar.
    """
    print("Generating Similarity Heatmap...")
    
    # Get topic embeddings
    topic_info = topic_model.get_topic_info()
    # Filter top N topics (excluding outlier)
    top_indices = topic_info[topic_info['Topic'] != -1].head(top_n)['Topic'].tolist()
    
    # Extract embeddings for these specific topics
    # Note: topic_embeddings_ is a list ordered by topic ID usually, 
    # but let's be safe and use the internal c-TF-IDF matrix if embeddings aren't cached
    
    if topic_model.topic_embeddings_ is not None:
        # Map topic IDs to embedding indices
        # This can be tricky in BERTopic versions. 
        # Safer way: calculate cosine sim of c-TF-IDF for the selected topics
        pass
    
    # EASIER METHOD: Use BERTopic's internal similarity matrix calculation
    # but implement plotting manually for better style
    
    # We will manually compute similarity between the topic words (c-TF-IDF)
    # topic_model.c_tfidf_ is a sparse matrix (n_topics, n_words)
    
    # Get the row indices corresponding to our top topics
    # topic_model.topic_mapper_ maps Topic ID -> Matrix Row Index
    # But usually Topic ID + 1 = Matrix Row Index (since -1 is at 0)
    # Let's trust the library's built-in helper if possible, or build a simple matrix
    
    try:
        # c-TF-IDF matrix
        tfidf = topic_model.c_tfidf_
        
        # Get indices for top_n topics. 
        # The topic_info['Topic'] is the actual topic ID.
        # We need to find where they are in the sparse matrix.
        # Usually, the model stores a mapping, or we can just iterate.
        # Let's rely on the text content for a safe visualization if matrix logic is opaque.
        
        # ACTUALLY, let's just use the embeddings if available, 
        # if not, we skip this plot to avoid errors.
        if topic_model.topic_embeddings_ is not None:
            # Filter embeddings for top_n topics
            # This assumes topic_embeddings_ is aligned with topic_info
            # We will skip the complexity and plot a simple correlation of the TOP WORDS
            pass
            
    except Exception as e:
        print(f"Skipping heatmap due to data access complexity: {e}")
        return

    # If we are here, let's just use the built-in visualize_heatmap data if we could,
    # but since we want "Beautiful Static Plots", let's create a visual 
    # based on the top 10 topics' cosine similarity
    
    # Simplify:
    matrix = cosine_similarity(topic_model.c_tfidf_)
    
    # The matrix includes the outlier at index 0 or similar.
    # Let's just grab the top N rows/cols
    # topic_info sorted by count gives us the IDs we want.
    
    target_ids = top_indices # e.g. [0, 1, 2...]
    # We need to map these IDs to the matrix indices.
    # In BERTopic, row index = topic_id + 1 (usually, because -1 is index 0)
    # Let's verify:
    
    valid_indices = [tid + 1 for tid in target_ids if (tid + 1) < matrix.shape[0]]
    
    if not valid_indices:
        return

    sub_matrix = matrix[np.ix_(valid_indices, valid_indices)]
    
    plt.figure(figsize=(10, 8))
    labels = [f"Topic {tid}" for tid in target_ids]
    
    sns.heatmap(sub_matrix, xticklabels=labels, yticklabels=labels, 
                cmap="RdBu_r", annot=True, fmt=".2f", vmin=0, vmax=1)
    
    plt.title(f"Semantic Similarity of Top {top_n} Topics", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = PLOTS / "topics_similarity.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close()

def main():
    print(f"Loading model from {MODELS}...")
    try:
        topic_model = BERTopic.load(MODELS / "bertopic_model")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # 1. Plot Topic Keywords (The "What")
    plot_top_keywords(topic_model, top_n=8)

    # 2. Plot Topic Sizes (The "How Many")
    plot_topic_sizes(topic_model, top_n=15)

    # 3. Plot Similarity (The "Relationships")
    # Note: Requires c-TF-IDF matrix which is standard in the model
    plot_similarity_heatmap(topic_model, top_n=10)
    
    print("\n✅ Visualization Complete. Check the 'plots' folder.")

if __name__ == "__main__":
    main()