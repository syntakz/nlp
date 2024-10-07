import streamlit as st
import pandas as pd
from nlp_lib import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sentences = [
    "Saya suka belajar data science.",
    "Python adalah bahasa pemrograman yang populer.",
    "Saya menggunakan Python untuk analisis data.",
    "Analisis data membantu dalam pengambilan keputusan.",
    "Machine learning adalah cabang dari kecerdasan buatan.",
    "Algoritma machine learning bisa memprediksi hasil.",
    "Saya tertarik pada teknologi baru.",
    "Kecerdasan buatan memiliki banyak aplikasi.",
    "Belajar data science sangat menarik.",
    "Saya mengikuti kursus online tentang machine learning."
]


# with st.sidebar:
#     st.write("Menu")

st.title("💬 Search Similarity")
# st.caption("🚀 Mencari Kata yang similar")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Masukkan kata/kalimat yang ingin dicari !"}]

if 'pdf_ref' not in st.session_state:
    st.session_state.pdf_ref = None

kalimat = [nlp.preprocessing(sentence) for sentence in sentences]
st.write(kalimat)



query = "Python"

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(sentences)

qry_vector = vectorizer.transform([query])
cosine_similarities = cosine_similarity(qry_vector, X_tfidf)
similarity_scores = cosine_similarities[0]
sorted_indices = np.argsort(similarity_scores)[::-1]

top_5_indices = sorted_indices[:5]
st.write("Top 5 sentences similar to 'data':")
for index in top_5_indices:
    st.write(f"Document {index + 1}: {sentences[index]} (Similarity score: {similarity_scores[index]:2f})")



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
