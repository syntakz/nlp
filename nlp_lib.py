import string 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv


class nlp:
    def preprocessing(data):
        vdata = data.lower()
        vdata = vdata.translate(vdata.maketrans("","", string.punctuation))
        tokens = word_tokenize(vdata)
        
        # nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

        ps = PorterStemmer()
        stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

        lemmatizer = WordNetLemmatizer()
        lemmatizer_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

        pos_tagged_tokens1 = pos_tag(stemmed_tokens)
        pos_tagged_tokens2 = pos_tag(lemmatizer_tokens)

        return '' .join(lemmatizer_tokens) 
        

    def extract_data(sentences, query):
        vectorizer = TfidfVectorizer()
        X_tfidf = vectorizer.fit_transform(sentences)

        qry_vector = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(qry_vector, X_tfidf)
        similarity_scores = cosine_similarities[0]
        sorted_indices = np.argsort(similarity_scores)[::-1]

        top_5_indices = sorted_indices[:5]
        content = [f"Document {index + 1}: {sentences[index]} (Similarity score: {similarity_scores[index]:.2f})" for index in top_5_indices]
        
        return content



