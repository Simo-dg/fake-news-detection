# utils_data.py
from pathlib import Path
import pandas as pd

def load_true_fake(path_true: str, path_fake: str) -> pd.DataFrame:
    true = pd.read_csv(path_true)
    fake = pd.read_csv(path_fake)

    def norm(df):
        cols = {c.lower().strip(): c for c in df.columns}
        title = df[cols['title']] if 'title' in cols else None
        text  = df[cols['text']]  if 'text'  in cols else None
        if title is not None and text is not None:
            combined = (title.fillna('') + " \n\n " + text.fillna('')).astype(str)
        elif text is not None:
            combined = text.fillna('').astype(str)
        else:
            combined = df.astype(str).agg(" ".join, axis=1)
        return combined

    true['text'] = norm(true)
    fake['text'] = norm(fake)
    true['label'] = 0
    fake['label'] = 1

    df = pd.concat([true[['text','label']], fake[['text','label']]], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
