import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
rf_model = joblib.load('fake_news_rf_model.pkl')
rf_vectorizer = joblib.load('tfidf_vectorizer_rf.pkl') 

# Preprocessing functions
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # Remove punctuation and lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalnum()]  # Remove stopwords and lemmatize
    return " ".join(tokens)



# Streamlit UI
st.title("Fake News Detection App")

# User input
# User input with an actual true label for comparison
user_input = st.text_area("Enter the news statement:")
true_label = st.selectbox("Select the actual label:", ['True', 'Mostly True', 'Half True', 'Barely True', 'False', 'Pants on Fire'])

if st.button("Predict"):
    if user_input:
        # Preprocess input
        processed_text = preprocess_text(user_input)
        rf_text_features = rf_vectorizer.transform([processed_text])
        rf_prediction_proba = rf_model.predict_proba(rf_text_features)
        
        # Get the predicted class (index) and its probability
        predicted_class = rf_model.predict(rf_text_features)[0]
        predicted_class_probability = rf_prediction_proba[0][predicted_class]

        # Define the label mapping
        label_mapping = {
            0: 'True',
            1: 'Mostly True',
            2: 'Half True',
            3: 'Barely True',
            4: 'False',
            5: 'Pants on Fire'
        }
        
        # Display the prediction and the confidence (probability)
        st.write(f"Prediction: {label_mapping[predicted_class]}")
        st.write(f"Confidence of prediction: {(1-predicted_class_probability) * 100:.2f}%")
    else:
        st.write("Please enter a news statement.")