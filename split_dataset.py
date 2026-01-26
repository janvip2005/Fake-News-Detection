import os
import sys
import pandas as pd

candidates = [
    "news-checkpoint.csv",
    os.path.join(".ipynb_checkpoints", "news-checkpoint.csv"),
    os.path.join("..", ".ipynb_checkpoints", "news-checkpoint.csv"),
]

def find_csv(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

src = find_csv(candidates)
if src is None:
    print("Could not find news-checkpoint.csv in project root or .ipynb_checkpoints.")
    print("Please place the file in the project root or .ipynb_checkpoints and re-run this script.")
    sys.exit(2)

print(f"Loading CSV from: {src}")

df = pd.read_csv(src)
if not {'title','text','label'}.issubset(df.columns):
    print("CSV missing required columns 'title', 'text', or 'label'. Columns found:", df.columns.tolist())
    sys.exit(3)

# normalize label and split
(df['label']) = df['label'].astype(str).str.strip()
real_df = df[df['label'].str.upper() == 'REAL']
fake_df = df[df['label'].str.upper() == 'FAKE']

real_out = 'news_real.csv'
fake_out = 'news_fake.csv'
real_df.to_csv(real_out, index=False)
fake_df.to_csv(fake_out, index=False)

print(f"Wrote {len(real_df)} REAL rows to {real_out}")
print(f"Wrote {len(fake_df)} FAKE rows to {fake_out}")
print('Done.')
