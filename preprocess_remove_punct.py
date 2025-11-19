import pandas as pd
import string
import sys

# Usage: python preprocess_remove_punct.py input.parquet output.parquet

def remove_punctuation(text):
    if not isinstance(text, str):
        return text
    return text.translate(str.maketrans('', '', string.punctuation))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_remove_punct.py input.parquet output.parquet")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(in_path)
    if 'text' in df.columns:
        df['text'] = df['text'].apply(remove_punctuation)
    else:
        print("Colonna 'text' non trovata nel dataset.")
        sys.exit(2)
    df.to_parquet(out_path)
    print(f"Salvato: {out_path}")
