from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from utils_data import load_true_fake
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512

def main():
    # Carica il dataset
    df = load_true_fake(DATA/"True.csv", DATA/"Fake.csv")
    
    # Carica il tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Calcola la lunghezza in token per ogni testo
    print("Tokenizing texts...")
    token_lengths = []
    for text in df['text']:
        tokens = tok(text, truncation=False, add_special_tokens=True)
        token_lengths.append(len(tokens['input_ids']))
    
    df['token_length'] = token_lengths
    
    # Statistiche
    total_samples = len(df)
    over_512 = df[df['token_length'] > MAX_LENGTH]
    num_over_512 = len(over_512)
    percentage_over_512 = (num_over_512 / total_samples) * 100
    
    print("\n" + "="*70)
    print("ANALISI LUNGHEZZA TOKEN NEL DATASET")
    print("="*70)
    print(f"Totale campioni: {total_samples:,}")
    print(f"Campioni con > {MAX_LENGTH} token: {num_over_512:,} ({percentage_over_512:.2f}%)")
    print(f"Campioni con <= {MAX_LENGTH} token: {total_samples - num_over_512:,} ({100-percentage_over_512:.2f}%)")
    print()
    print(f"Lunghezza minima: {df['token_length'].min()} token")
    print(f"Lunghezza massima: {df['token_length'].max()} token")
    print(f"Lunghezza media: {df['token_length'].mean():.2f} token")
    print(f"Lunghezza mediana: {df['token_length'].median():.0f} token")
    print(f"Deviazione standard: {df['token_length'].std():.2f} token")
    print()
    
    if num_over_512 > 0:
        print("STATISTICHE PER CAMPIONI CHE SUPERANO 512 TOKEN:")
        print("-" * 70)
        excess_tokens = over_512['token_length'] - MAX_LENGTH
        print(f"Token in eccesso - minimo: {excess_tokens.min()} token")
        print(f"Token in eccesso - massimo: {excess_tokens.max()} token")
        print(f"Token in eccesso - media: {excess_tokens.mean():.2f} token")
        print(f"Token in eccesso - mediana: {excess_tokens.median():.0f} token")
        print()
        
        # Percentili
        print("DISTRIBUZIONE TOKEN IN ECCESSO:")
        print("-" * 70)
        for percentile in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(excess_tokens, percentile)
            print(f"{percentile}° percentile: {val:.0f} token in eccesso")
    
    print("="*70)
    
    # Salva risultati in CSV
    results_df = df[['token_length']].copy()
    results_df['over_512'] = results_df['token_length'] > MAX_LENGTH
    results_df['excess_tokens'] = results_df['token_length'] - MAX_LENGTH
    results_df['excess_tokens'] = results_df['excess_tokens'].apply(lambda x: max(0, x))
    results_df.to_csv(PLOTS / "token_length_analysis.csv", index=False)
    print(f"\nRisultati salvati in: {PLOTS / 'token_length_analysis.csv'}")
    
    # Crea istogramma
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Distribuzione completa
    plt.subplot(1, 2, 1)
    plt.hist(df['token_length'], bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(x=MAX_LENGTH, color='r', linestyle='--', linewidth=2, label=f'Max length ({MAX_LENGTH})')
    plt.xlabel('Numero di token')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione lunghezza token')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Focus su campioni > 512
    if num_over_512 > 0:
        plt.subplot(1, 2, 2)
        plt.hist(over_512['token_length'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.axvline(x=MAX_LENGTH, color='r', linestyle='--', linewidth=2, label=f'Max length ({MAX_LENGTH})')
        plt.xlabel('Numero di token')
        plt.ylabel('Frequenza')
        plt.title(f'Distribuzione per campioni > {MAX_LENGTH} token')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS / "token_length_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {plot_path}")
    plt.close()
    
    # Crea grafico per token in eccesso
    if num_over_512 > 0:
        plt.figure(figsize=(10, 6))
        excess_tokens = over_512['token_length'] - MAX_LENGTH
        plt.hist(excess_tokens, bins=50, edgecolor='black', alpha=0.7, color='red')
        plt.xlabel('Token in eccesso oltre 512')
        plt.ylabel('Frequenza')
        plt.title(f'Distribuzione token in eccesso per {num_over_512:,} campioni ({percentage_over_512:.2f}%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        excess_plot_path = PLOTS / "excess_tokens_distribution.png"
        plt.savefig(excess_plot_path, dpi=300, bbox_inches='tight')
        print(f"Grafico token in eccesso salvato in: {excess_plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
