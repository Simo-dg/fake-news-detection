import matplotlib
matplotlib.use("Agg") # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
import config

# --- CONFIGURATION ---
MODELS = config.MODELS_DIR
PLOTS = config.PLOTS_DIR

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
    
    print("\n✅ Visualization Complete. Check the 'plots' folder.")

if __name__ == "__main__":
    main()