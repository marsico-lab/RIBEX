import pickle
import os
from pathlib import Path

REPO = Path(os.getenv("REPOSITORY", "/path/to/RBP_IG_storage"))
DATA_SETS = REPO / 'data' / 'data_sets'
EMBEDDINGS = REPO / 'data' / 'embeddings'

ds_path = DATA_SETS / 'RIC_2_human_fine-tuning.pkl'
with open(ds_path, 'rb') as f:
    df = pickle.load(f)

total_genes = set(df['Gene_ID'])
emb_folder = EMBEDDINGS / 'protT5_xl_uniref50' / 'RIC'
existing = set(p.name for p in emb_folder.iterdir())
missing = total_genes - existing

df_kept = df[~df['Gene_ID'].isin(missing)]
df_dropped = df[df['Gene_ID'].isin(missing)]

print(f'LoRA uses:    {len(df)} samples ({len(total_genes)} genes)')
print(f'FiLM_PE uses: {len(df_kept)} samples ({len(total_genes - missing)} genes)')
print(f'Dropped:      {len(df_dropped)} samples ({len(missing)} genes)')
print(f'Pos rate full:    {df["positive"].mean():.4f}')
print(f'Pos rate kept:    {df_kept["positive"].mean():.4f}')
if len(df_dropped) > 0:
    print(f'Pos rate dropped: {df_dropped["positive"].mean():.4f}')
    for g in sorted(missing):
        sub = df[df['Gene_ID'] == g]
        print(f'  {g}: {len(sub)} samples, pos={sub["positive"].mean():.3f}')
        print(f'  {g}: {len(sub)} samples, pos={sub["positive"].mean():.3f}')
