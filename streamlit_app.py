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





for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    content = nlp.extract_data(sentences,prompt)
    msg = content
    st.write("Similarity kata " + prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
