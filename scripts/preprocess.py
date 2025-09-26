import os 
import pandas as pd

#Paths
RAW_DIR = "data/raw"
SAMPLE_FILE = "data/sample/sample_reviews.csv"
PROCESSED_FILE = "data/processed/cleaned_reviews.csv"

def load_data():
  """
  Load Dataset.
  - If full dataset exists in data/raw, use it.
  - Otherwise, fall back to sample dataset.
  """
  #Check if raw folder has any file
  raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith((".csv", ".json"))]
  if raw_files:
    raw_path = os.path.join(RAW_DIR, raw_files[0])
    print(f"Using full dataset: {raw_path}")
    if raw_path.endswith(".csv"):
      return pd.read_csv(raw_path)
    elif raw_path.endswith(".json"):
      return pd.read_json(raw_path, lines=True)

  else:
    print(f"No raw dataset found. Falling back to sample: {SAMPLE_FILE}")
    return pd.read_csv(SAMPLE_FILE)

def preprocess(df: pd.DataFrame):
  """
  Clean and standardise the dataset for downstream analysis.
  """
  #Normalise column names
  df.columns = [c.lower().replace(" ", "_") for c in df.columns

  #Keep relevant fields
  expected_cols = ["review_id", "product_name", "rating", "review_text", "review_date"]
  df = df[[c for c is expected_cols if c in df.columns]]
  
  #Drop rows with missing text
  df = df.dropna(subset=["review_text"])
  
  #Simple cleaning of text
  df["review_text"] = (
       df["review_text"]
       .astype(str)
       .str.strip()
       .str.replace(r"s+", " ", regex=True)
  )
  #Ensure rating is numeric
  if"rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

  #Drop dulicates
  df = df.drop_duplicates(subset=["review_id", "review_text"])

  return df

def main():
  df = load_data()
  print(f"Loaded {len(df)} rows")

  cleaned = preprocess(df)
  print(f"Cleaned dataset: {len(cleaned)} rows")

  #Save to proprocessed folder
  os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
  cleaned.to_csv(PROCESSED_FILE, index=False)

  print(f"Saved cleaned data -> {PROCESSED_FILE}")

if __name__ == "__main__":
  main()

  
  
