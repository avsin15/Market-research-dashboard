"""
Step 4: Generative AI Summaries (Gemini version)
------------------------------------------------
- Loads dataset with topics + sentiment
- Groups reviews by topic
- Calls Google Gemini to summarize reviews per topic
- Saves outputs into JSON and augments CSV with summaries

Usage:
    export GOOGLE_API_KEY="your_api_key_here"
    python scripts/genai_summaries_gemini.py
"""

import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

#Load environment variables from .env
load_dotenv()

#api_key = os.getenv("GOOGLE_API_KEY")
#if not api_key:
  #  raise RuntimeError("❌ GOOGLE_API_KEY not found. Set it before running.")

#genai.configure(api_key=api_key)

# Paths
SENTIMENT_FILE = "data/sample/reviews_with_sentiment.csv"
SUMMARY_JSON_FILE = "data/sample/topic_summaries.json"
SUMMARY_CSV_FILE = "data/sample/reviews_with_summaries.csv"

# Configure Gemini (needs GOOGLE_API_KEY in environment)
genai.configure(api_key= os.getenv("Your API here"))


def generate_summary(topic_id, reviews, max_samples=50):
    """
    Generate a summary for a topic using Gemini.
    - Samples up to `max_samples` reviews to stay within token/token limit.
    """
    sample_reviews = reviews[:max_samples]

    prompt = f"""
    You are a market research assistant.
    Summarize customer feedback for Topic {topic_id}.
    Highlight:
    - Main themes
    - Positive aspects
    - Negative aspects
    - Suggestions for improvement

    Here are {len(sample_reviews)} sample reviews:
    {sample_reviews}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")  # Fast + cost-efficient, switched to gemini-2.5-flash
    response = model.generate_content(prompt)

    return response.text.strip() if response and response.text else "No Summary Generated."


def main():
    # Load Dataset 
    if not os.path.exists(SENTIMENT_FILE): 
        raise FileNotFoundError(f"❌ Missing file: {SENTIMENT_FILE}. Run sentiment_analysis.py first.")

    df = pd.read_csv(SENTIMENT_FILE)
    print(f"✅ Loaded {len(df)} reviews with topics + sentiment")

    # Group by topic
    topic_groups = df.groupby("topic")["review_text"].apply(list).to_dict()

    summaries = {}
    for topic_id, reviews in topic_groups.items():
        print(f"🔄 Generating summary for Topic {topic_id} ({len(reviews)} reviews)...")
        try:
            summary = generate_summary(topic_id, reviews)
            summaries[topic_id] = summary
        except Exception as e:
            print(f"⚠️ Failed to summarize Topic {topic_id}: {e}")
            summaries[topic_id] = "Error generating summary."

    # Save JSON
    os.makedirs(os.path.dirname(SUMMARY_JSON_FILE), exist_ok=True)
    with open(SUMMARY_JSON_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"💾 Saved summaries → {SUMMARY_JSON_FILE}")

    # Add summaries to dataset
    df["topic_summary"] = df["topic"].map(summaries)
    df.to_csv(SUMMARY_CSV_FILE, index=False)
    print(f"💾 Saved augmented dataset → {SUMMARY_CSV_FILE}")


if __name__ == "__main__":
    main()
