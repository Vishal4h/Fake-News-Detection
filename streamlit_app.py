import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load trained model & vectorizer
with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# ----------------- UI Enhancements -----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown('<h1 style="text-align:center;">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center; color:#555;">Enter news text or upload a file to check if it\'s **Fake** or **Real**.</h3>', unsafe_allow_html=True)
st.write("---")  # Horizontal line

# Text input section
user_input = st.text_area("✍️ Enter News Text:", height=150)

# File upload section
uploaded_file = st.file_uploader("📂 Upload a .txt or .csv file for bulk prediction", type=["txt", "csv"])

# ----------------- Prediction -----------------
if st.button("🔍 Predict"):
    with st.spinner("Analyzing the news... ⏳"):  # Show loading animation
        if user_input:
            # Predict for single input
            cleaned_text = clean_text(user_input)
            transformed_text = tfidf_vectorizer.transform([cleaned_text])
            prediction_proba = model.predict_proba(transformed_text)[0]  # Get probabilities

            fake_prob = prediction_proba[0]  # Probability of Fake
            real_prob = prediction_proba[1]  # Probability of Real
            confidence_score = max(fake_prob, real_prob) * 100  # Convert to percentage

            if real_prob > fake_prob:
                st.success(f"✅ **Real News** - {confidence_score:.2f}% confidence")
            else:
                st.error(f"❌ **Fake News** - {confidence_score:.2f}% confidence")

        elif uploaded_file is not None:
            # Handle uploaded file
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode("utf-8")
                news_articles = content.split("\n")  # Split text into lines
                df = pd.DataFrame(news_articles, columns=["News Article"])
    
                # ✅ Ensure "Cleaned News" column exists
                df["Cleaned News"] = df["News Article"].apply(clean_text)

            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)

                # Automatically find the correct text column
                text_col = None
                for col in df.columns:
                    if df[col].dtype == object and df[col].str.len().mean() > 20:  # Check if column has text
                        text_col = col
                        break

                if text_col:
                    df["Cleaned News"] = df[text_col].apply(clean_text)
                else:
                    st.error("❌ Could not detect a valid news text column in the CSV file. Please check your file format.")
                st.stop()

            df.dropna(inplace=True)  # Remove empty rows
            transformed_data = tfidf_vectorizer.transform(df["Cleaned News"])
            predictions_proba = model.predict_proba(transformed_data)  # Get probability scores
            predictions = model.predict(transformed_data)

            # Add confidence score to dataframe
            df["Confidence (%)"] = (predictions_proba.max(axis=1) * 100).round(2)
            df["Prediction"] = ["✅ Real" if pred == 1 else "❌ Fake" for pred in predictions]

            st.write("### 📊 Bulk Prediction Results (With Confidence Scores):")
            st.dataframe(df[["Prediction", "Confidence (%)"]], use_container_width=True)

            # Download results as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")

        else:
            st.warning("⚠️ Please enter text or upload a file before clicking 'Predict'.")
        