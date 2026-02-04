import streamlit as st
import pickle
import re
import os

# -------------------------------------------------
# NLTK SAFE INIT (THIS MUST BE FIRST)
# -------------------------------------------------
nltk.download("punkt", quiet=True)

from nltk.stem import PorterStemmer

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Food Review Sentiment Analyzer",
    layout="wide"
)

# -------------------------------------------------
# Load Models (RELATIVE PATH ONLY)
# -------------------------------------------------
MODEL_DIR = "models"

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# NLP Preprocessing
# -------------------------------------------------
ps = PorterStemmer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# -------------------------------------------------
# Food Vocabulary
# -------------------------------------------------
FOOD_ITEMS = [
    "pizza", "burger", "biryani", "rice", "chicken",
    "paneer", "naan", "roti", "fries", "pasta",
    "sandwich", "food", "restaurant", "taste",
    "service", "delivery", "menu", "dosa", "idli",
    "samosa", "curry"
]

def extract_food_items(text):
    text = text.lower()
    return list(set([f for f in FOOD_ITEMS if f in text]))

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown("## 🍽️ Food Review Sentiment Analyzer")
st.write("TF-IDF + Random Forest NLP Model")

review = st.text_area("Enter your food review")
analyze = st.button("Analyze")

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if analyze:
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        foods = extract_food_items(review)

        if not foods:
            st.warning("This model only works for food-related reviews.")
        else:
            st.success(f"Detected food items: {', '.join(foods)}")

            cleaned = preprocess(review)
            X_new = tfidf.transform([cleaned])
            prediction = model.predict(X_new)[0]

            if prediction == 1:
                st.success("Positive Review 😊")
            else:
                st.error("Negative Review 😞")




