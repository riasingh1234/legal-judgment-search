
import streamlit as st
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Legal Judgment Search",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Indian Supreme Court Judgment Search")
st.write("Search judgments and extract relevant legal evidence.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,;:/()-]', '', text)
    return text.strip()

@st.cache_resource
def load_data():
    df = pd.read_csv("judgments_demo.csv")
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 100].reset_index(drop=True)

    tokenized = [text.split() for text in df["clean_text"]]
    bm25 = BM25Okapi(tokenized)

    return df, bm25

try:
    data, bm25 = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

st.success(f"Dataset loaded: {len(data)} judgments")

query = st.text_input(
    "Enter your legal query",
    placeholder="Example: Right to privacy under Article 21"
)

top_k = st.slider("Number of judgments", 1, 10, 5)

if st.button("Search Judgments") and query.strip():

    scores = bm25.get_scores(clean_text(query).split())
    indices = scores.argsort()[::-1][:top_k]

    st.subheader("Relevant Judgments")

    for rank, idx in enumerate(indices, 1):

        row = data.iloc[idx]

        with st.expander(
            f"{rank}. {row.get('number', 'Case')} | "
            f"BM25 Score: {scores[idx]:.3f}",
            expanded=(rank == 1)
        ):

            st.write("**Court:**", row.get("court", "N/A"))
            st.write("**Date:**", row.get("date", "N/A"))
            st.write("**Judge:**", row.get("in_judge", "N/A"))
            st.write("**Petitioner:**", row.get("in_petitioner", "N/A"))
            st.write("**Respondent:**", row.get("in_respondent", "N/A"))

            text = str(row["text"])

            paragraphs = [
                p.strip() for p in re.split(r'\n+|(?<=[.!?])\s+', text)
                if len(p.strip()) > 80
            ]

            if paragraphs:
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=10000
                )

                try:
                    matrix = vectorizer.fit_transform(
                        [query] + paragraphs
                    )

                    similarities = cosine_similarity(
                        matrix[0:1], matrix[1:]
                    ).flatten()

                    top = similarities.argsort()[::-1][:3]

                    st.markdown("**Relevant Evidence:**")

                    for i in top:
                        if similarities[i] > 0:
                            st.write(
                                f"Similarity: {similarities[i]:.4f}"
                            )
                            st.info(paragraphs[i][:1500])

                except ValueError:
                    st.write("No matching evidence found.")

            articles = re.findall(
                r'\bArticles?\s+\d+[A-Z]?',
                text,
                re.I
            )

            sections = re.findall(
                r'\bSections?\s+\d+[A-Z]?',
                text,
                re.I
            )

            references = re.findall(
                r'\b\d{4}\s+INSC\s+\d+\b',
                text,
                re.I
            )

            st.markdown("**Extracted Legal Information**")
            st.write("Articles:", list(set(articles))[:20])
            st.write("Sections:", list(set(sections))[:20])
            st.write("Case References:", list(set(references))[:20])

elif st.button("Search Judgments"):
    st.warning("Please enter a legal query.")

st.divider()
st.caption("Academic prototype | BM25 | TF-IDF | Cosine Similarity | Regex")
