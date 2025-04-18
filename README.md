# 🎬 Movie Review Sentiment Analyzer

A simple yet effective web app that predicts whether a movie review expresses **positive** or **negative** sentiment using a Naive Bayes machine learning model. Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🌟 Features

- 📥 Accepts custom movie reviews as text input  
- 🔍 Predicts sentiment (Positive or Negative)  
- 📊 Displays model confidence  
- 🤖 Trained on the NLTK Movie Reviews Dataset  
- 💻 Easy-to-use interactive Streamlit web interface  

---


## 🧠 How It Works

1. The model uses **TF-IDF Vectorization** to process text.
2. A **Naive Bayes** classifier predicts sentiment based on review content.
3. Model and vectorizer are saved using `joblib` for reuse.

---

## 🛠️ Installation
```bash
 1. Clone the Repo
git clone https://github.com/yghodak/sentiment-analyzer.git
cd sentiment-analyzer

 2. Install Dependencies
pip install -r requirements.txt

 3. Train the Model (Optional)
python train_sentiment_model.py

 4.  Run the App
streamlit run app.py
```

---

## 📁 Project Structure
```bash
sentiment-analyzer/
├── app.py                  # Streamlit web app
├── train_sentiment_model.py # Script to train and save the model
├── sentiment_model.pkl     # Trained model
├── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt        # Dependencies
└── README.md
```

---

## 📦 Requirements
```bash
Python 3.x, streamlit, scikit-learn, pandas, nltk, joblib
```
---

## 🧑‍💻 Author

### Made by Yash Ghodake ✨

