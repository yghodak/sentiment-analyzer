import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords

# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and I'll predict if it's Positive or Negative!")

# User input
review = st.text_area("Enter Movie Review:")

# Predict button
if st.button("🔍 Predict Sentiment"):
    if review:
        # Preprocess and vectorize review
        review_tfidf = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        if prediction == 'pos':
            st.success("👍 Positive Sentiment!")
        else:
            st.error("👎 Negative Sentiment!")
    else:
        st.warning("Please enter a review!")
