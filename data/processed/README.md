# Processed Data Folder

This folder stores cleaned, preprocessed, or feature-engineered datasets
(e.g., embeddings, clustering outputs).

It is **ignored by Git** to keep the repo small.

Each user can regenerate these files by running the pipeline:
```bash
python scripts/preprocess.py
