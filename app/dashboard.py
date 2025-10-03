
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gen-AI Market Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SUMMARY_FILE = "data/sample/reviews_with_summaries.csv"

@st.cache_data
def load_data():
    if not os.path.exists(SUMMARY_FILE):
        st.error("❌ Missing data. Run genai_summaries_gemini.py first.")
        st.stop()
    df = pd.read_csv(SUMMARY_FILE)
    df['sentiment'] = df['sentiment'].astype(str)
    df['topic'] = df['topic'].astype(str)
    return df

df = load_data()

st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("### Filters")

topic_options = ["All Topics"] + sorted(df["topic"].unique().tolist())
topic_filter = st.sidebar.selectbox("Select a Topic", topic_options)

sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=sorted(df["sentiment"].unique()),
    default=list(df["sentiment"].unique())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Search")
search_text = st.sidebar.text_input("🔍 Search reviews", placeholder="Enter keywords...")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Overview")
st.sidebar.metric("Total Reviews", len(df))
st.sidebar.metric("Topics", df["topic"].nunique())
st.sidebar.metric("Avg Review Length", f"{df['review_text'].str.len().mean():.0f} chars")

st.title("📊 Gen-AI Market Research Dashboard")
st.caption("Explore topics, sentiments, and AI-generated insights from customer reviews")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reviews", len(df))
with col2:
    positive_pct = (df["sentiment"].str.lower().str.contains("positive").sum() / len(df) * 100)
    st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
with col3:
    st.metric("Unique Topics", df["topic"].nunique())
with col4:
    if "rating" in df.columns:
        avg_rating = df["rating"].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}")
    else:
        st.metric("Data Points", len(df))

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📌 Topic Distribution")
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]

    fig = px.bar(
        topic_counts, 
        x="count", 
        y="topic",
        orientation='h',
        title="Number of Reviews per Topic",
        color="count",
        color_continuous_scale="Blues"
    )
    fig.update_layout(showlegend=False, height=400, xaxis_title="Number of Reviews", yaxis_title="Topic")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("😊 Sentiment Distribution")
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]

    fig2 = px.pie(
        sent_counts, 
        values="count", 
        names="sentiment",
        title="Overall Sentiment Breakdown",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("🔥 Sentiment Distribution Across Topics")
sentiment_topic = pd.crosstab(df['topic'], df['sentiment'])
fig3 = px.imshow(
    sentiment_topic,
    labels=dict(x="Sentiment", y="Topic", color="Count"),
    title="Sentiment Heatmap by Topic",
    color_continuous_scale="YlOrRd"
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

if topic_filter == "All Topics":
    st.subheader("🔍 All Topics Overview")
    topic_data = df[df["sentiment"].isin(sentiment_filter)]

    st.markdown("### 🤖 AI-Generated Topic Summaries")
    for topic in sorted(df["topic"].unique()):
        topic_subset = df[df["topic"] == topic]
        if not topic_subset.empty and "topic_summary" in topic_subset.columns:
            summary_text = topic_subset["topic_summary"].iloc[0]
            with st.expander(f"📋 Topic {topic}"):
                st.write(summary_text)
else:
    st.subheader(f"🔍 Topic {topic_filter} Deep Dive")
    topic_data = df[(df["topic"] == topic_filter) & (df["sentiment"].isin(sentiment_filter))]

    col1, col2 = st.columns([2, 1])

    with col1:
        if not topic_data.empty and "topic_summary" in topic_data.columns:
            summary_text = topic_data["topic_summary"].iloc[0]
            with st.expander("🤖 Gemini-Generated Topic Summary", expanded=True):
                st.write(summary_text)

    with col2:
        topic_sent = topic_data["sentiment"].value_counts().reset_index()
        topic_sent.columns = ["sentiment", "count"]
        fig4 = px.pie(topic_sent, values="count", names="sentiment", title=f"Sentiment for Topic {topic_filter}", hole=0.3)
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

if search_text:
    topic_data = topic_data[topic_data["review_text"].str.contains(search_text, case=False, na=False)]
    st.info(f"🔍 Found {len(topic_data)} reviews matching '{search_text}'")

st.markdown("---")
st.subheader("📝 Review Details")

if topic_data.empty:
    st.warning("No reviews match the current filters.")
else:
    sort_option = st.selectbox("Sort reviews by:", ["Original Order", "Sentiment", "Review Length"], key="sort_reviews")

    if sort_option == "Sentiment":
        topic_data = topic_data.sort_values("sentiment")
    elif sort_option == "Review Length":
        topic_data = topic_data.assign(review_length=topic_data["review_text"].str.len())
        topic_data = topic_data.sort_values("review_length", ascending=False)

    reviews_per_page = st.slider("Reviews per page", 5, 50, 20)
    total_reviews = len(topic_data)
    total_pages = (total_reviews - 1) // reviews_per_page + 1

    page = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * reviews_per_page
    end_idx = start_idx + reviews_per_page

    st.write(f"Showing reviews {start_idx + 1}-{min(end_idx, total_reviews)} of {total_reviews}")

    for idx, row in topic_data.iloc[start_idx:end_idx].iterrows():
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**Review #{idx}**")
                st.write(row["review_text"])
            with col2:
                sentiment_emoji = "😊" if "positive" in str(row["sentiment"]).lower() else "😐" if "neutral" in str(row["sentiment"]).lower() else "😞"
                st.markdown(f"**{sentiment_emoji} {row['sentiment']}**")
            st.markdown("---")

    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=topic_data.to_csv(index=False).encode('utf-8'),
        file_name=f"filtered_reviews_topic_{topic_filter}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Dashboard powered by Streamlit, Plotly, and Google Gemini AI")

# =========================
# Ask the Dataset (Gemini)
# =========================
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.subheader("🤖 Ask the Dataset")

# Input box for user query
user_query = st.text_area("Enter your question about customer feedback:")

# Configure Gemini (needs GOOGLE_API_KEY in environment)
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key= os.getenv("AIzaSyAIxlvhkYSS5SB0z5YB8bUgD853hvFwjBA"))
model = genai.GenerativeModel("gemini-2.5-flash")

if st.button("Get Answer") and user_query:
    # Create context: compress dataset into a prompt
    sample_data = df[["review_text", "sentiment", "topic"]].head(100).to_dict(orient="records")

    prompt = f"""
    You are a market research analyst.
    Use the following dataset of customer reviews (with topics and sentiment)
    to answer the question:

    Question: {user_query}

    Dataset (sample of {len(sample_data)} reviews):
    {sample_data}

    Please provide a concise, data-driven answer highlighting patterns and insights.
    """

    with st.spinner("Thinking with Gemini..."):
        try:
            response = model.generate_content(prompt)
            st.success("Answer:")
            st.write(response.text if response and response.text else "No response generated.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

