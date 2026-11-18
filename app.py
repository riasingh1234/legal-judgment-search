
import streamlit as st
import pandas as pd
import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Legal Judgment Search",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Indian Supreme Court Judgment Search")
st.write(
    "Search relevant Supreme Court judgments and extract "
    "important evidence from legal documents."
)


# ---------------- TEXT PREPROCESSING ----------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,;:/()-]', '', text)
    return text.strip()


# ---------------- LOAD DATASET ----------------

@st.cache_resource
def load_data():

    data = pd.read_csv("judgments_demo.csv")

    data = data.dropna(subset=["text"]).copy()

    data["text"] = data["text"].astype(str)

    data["clean_text"] = data["text"].apply(clean_text)

    data = data[data["clean_text"].str.len() > 100]

    data = data.reset_index(drop=True)

    tokenized_docs = [
        text.split() for text in data["clean_text"]
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return data, bm25


try:
    data, bm25 = load_data()

    st.success(
        f"Dataset loaded successfully! "
        f"{len(data)} judgments available."
    )

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ---------------- EVIDENCE EXTRACTION ----------------

def extract_evidence(query, judgment_text, top_n=3):

    paragraphs = [
        p.strip()
        for p in re.split(r'\n+|(?<=[.!?])\s+', judgment_text)
        if len(p.strip()) > 80
    ]

    if not paragraphs:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    matrix = vectorizer.fit_transform(
        [query] + paragraphs
    )

    scores = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = []

    for i in top_indices:
        if scores[i] > 0:
            results.append(
                (paragraphs[i], round(float(scores[i]), 4))
            )

    return results


# ---------------- LEGAL INFORMATION EXTRACTION ----------------

def extract_legal_information(text):

    articles = re.findall(
        r'\bArticles?\s+\d+[A-Z]?(?:\s*\([^)]+\))?',
        text,
        re.IGNORECASE
    )

    sections = re.findall(
        r'\bSections?\s+\d+[A-Z]?',
        text,
        re.IGNORECASE
    )

    case_references = re.findall(
        r'\b\d{4}\s+INSC\s+\d+\b',
        text,
        re.IGNORECASE
    )

    return {
        "Articles": sorted(set(articles)),
        "Sections": sorted(set(sections)),
        "Case References": sorted(set(case_references))
    }


# ---------------- SEARCH INTERFACE ----------------

st.subheader("Search Legal Judgments")

query = st.text_input(
    "Enter your legal query:",
    placeholder="Example: Right to privacy under Article 21"
)

top_k = st.slider(
    "Number of judgments to retrieve",
    min_value=1,
    max_value=10,
    value=5
)


# ---------------- SEARCH BUTTON ----------------

if st.button("Search Judgments", key="search_button"):

    if not query.strip():

        st.warning("Please enter a legal query.")

    else:

        with st.spinner("Searching judgments..."):

            query_tokens = clean_text(query).split()

            scores = bm25.get_scores(query_tokens)

            indices = scores.argsort()[::-1][:top_k]

        st.subheader("Relevant Judgments")

        found_results = False

        for rank, idx in enumerate(indices, 1):

            if scores[idx] <= 0:
                continue

            found_results = True

            row = data.iloc[idx]

            with st.expander(
                f"{rank}. {row.get('number', 'Case number unavailable')} "
                f"| BM25 Score: {scores[idx]:.3f}",
                expanded=(rank == 1)
            ):

                st.markdown("### Case Details")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Document ID:**", row.get("doc_id", "N/A"))
                    st.write("**Court:**", row.get("court", "N/A"))
                    st.write("**Date:**", row.get("date", "N/A"))
                    st.write("**Case Number:**", row.get("number", "N/A"))

                with col2:
                    st.write("**Judge:**", row.get("in_judge", "N/A"))
                    st.write("**Petitioner:**", row.get("in_petitioner", "N/A"))
                    st.write("**Respondent:**", row.get("in_respondent", "N/A"))

                st.markdown("### Relevant Evidence")

                evidence = extract_evidence(
                    query,
                    row["text"],
                    top_n=3
                )

                if evidence:

                    for i, (paragraph, similarity) in enumerate(
                        evidence, 1
                    ):

                        st.markdown(
                            f"**Evidence {i} | Similarity Score: "
                            f"{similarity}**"
                        )

                        st.write(paragraph)

                        st.divider()

                else:

                    st.info("No matching evidence passages found.")

                st.markdown("### Extracted Legal Information")

                legal_info = extract_legal_information(
                    row["text"]
                )

                st.write(
                    "**Articles:**",
                    ", ".join(legal_info["Articles"]) or "None found"
                )

                st.write(
                    "**Sections:**",
                    ", ".join(legal_info["Sections"]) or "None found"
                )

                st.write(
                    "**Case References:**",
                    ", ".join(legal_info["Case References"]) or "None found"
                )

        if not found_results:

            st.warning(
                "No relevant judgments found. "
                "Try a different legal query."
            )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Academic Project | Indian Supreme Court Judgment Search "
    "and Evidence Extraction System"
)
