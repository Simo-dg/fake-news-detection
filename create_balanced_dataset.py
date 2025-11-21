"""
Create a balanced dataset with an equal number of FAKE and REAL articles.
"""
import pandas as pd
import glob
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"

# Configuration
TARGET_SIZE = 200_000  
TARGET_FAKE = TARGET_SIZE // 2  # 100k FAKE
TARGET_REAL = TARGET_SIZE // 2  # 100k REAL

print(f"{'='*80}")
print(f"CREATING BALANCED DATASET")
print(f"{'='*80}")
print(f"Target: {TARGET_SIZE:,} articles (50% FAKE, 50% REAL)")
print(f"{'='*80}\n")

# Carica tutti i file
parquet_files = sorted(glob.glob(str(DATA / "clean_df_part_*.parquet")))
print(f"Loading {len(parquet_files)} parquet files...\n")

all_fake = []
all_real = []

for f in parquet_files:
    print(f"  Loading {Path(f).name}...")
    df = pd.read_parquet(f)
    
    
    if 'y_fake' in df.columns:
        df = df.rename(columns={'y_fake': 'label'})
    
    # Separate FAKE and REAL
    fake_rows = df[df['label'] == 1]
    real_rows = df[df['label'] == 0]
    
    if len(fake_rows) > 0:
        all_fake.append(fake_rows)
    if len(real_rows) > 0:
        all_real.append(real_rows)
    
    print(f"    FAKE: {len(fake_rows):,} | REAL: {len(real_rows):,}")

# concatenate
print(f"\nConcatenating...")
df_fake = pd.concat(all_fake, ignore_index=True)
df_real = pd.concat(all_real, ignore_index=True)

print(f"  Total FAKE available: {len(df_fake):,}")
print(f"  Total REAL available: {len(df_real):,}")

# Remove duplicates before sampling
print(f"\nRemoving duplicates...")
before_fake = len(df_fake)
before_real = len(df_real)

df_fake = df_fake.drop_duplicates(subset=['title', 'text'], keep='first')
df_real = df_real.drop_duplicates(subset=['title', 'text'], keep='first')

print(f"  FAKE: {before_fake:,} → {len(df_fake):,} (removed {before_fake - len(df_fake):,})")
print(f"  REAL: {before_real:,} → {len(df_real):,} (removed {before_real - len(df_real):,})")

# Check if we have enough data
if len(df_fake) < TARGET_FAKE:
    print(f"\n⚠️  WARNING: Not enough FAKE articles ({len(df_fake):,} < {TARGET_FAKE:,})")
    print(f"   Using all available FAKE articles")
    TARGET_FAKE = len(df_fake)
    TARGET_REAL = TARGET_FAKE  # Keep balanced

if len(df_real) < TARGET_REAL:
    print(f"\n⚠️  WARNING: Not enough REAL articles ({len(df_real):,} < {TARGET_REAL:,})")
    print(f"   Using all available REAL articles")
    TARGET_REAL = len(df_real)
    TARGET_FAKE = TARGET_REAL  # Keep balanced

# Random sampling
print(f"\nSampling {TARGET_FAKE:,} FAKE + {TARGET_REAL:,} REAL...")
np.random.seed(42)

sampled_fake = df_fake.sample(n=TARGET_FAKE, random_state=42)
sampled_real = df_real.sample(n=TARGET_REAL, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([sampled_fake, sampled_real], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n{'='*80}")
print(f"BALANCED DATASET CREATED")
print(f"{'='*80}")
print(f"Total articles: {len(balanced_df):,}")
print(f"FAKE: {(balanced_df['label']==1).sum():,} ({(balanced_df['label']==1).sum()/len(balanced_df)*100:.1f}%)")
print(f"REAL: {(balanced_df['label']==0).sum():,} ({(balanced_df['label']==0).sum()/len(balanced_df)*100:.1f}%)")
print(f"{'='*80}\n")

# Save
output_file = DATA / "balanced_dataset_200k.parquet"
balanced_df.to_parquet(output_file, index=False)
print(f"✅ Saved to: {output_file}")

# Final statistics
print(f"\nDataset info:")
print(f"  Columns: {list(balanced_df.columns)}")
print(f"  Memory usage: {balanced_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")
print(f"\n{'='*80}")
print(f"✅ DONE! Use this file for training:")
print(f"   {output_file.name}")
print(f"{'='*80}\n")