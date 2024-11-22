# Import necessary libraries
import streamlit as st  # For building the web application
import pickle  # For loading pre-trained models and vectorizers
import string  # For handling punctuation
from nltk.corpus import stopwords  # For removing stopwords
import nltk  # For text preprocessing
from nltk.stem.porter import PorterStemmer  # For stemming words

# Initialize the PorterStemmer for stemming
ps = PorterStemmer()

# Function to preprocess and transform input text
def transform_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Tokenize the text into individual words
    text = nltk.word_tokenize(text)

    y = []  # Temporary list to store intermediate results

    # Remove non-alphanumeric tokens (e.g., punctuation and special characters)
    for i in text:
        if i.isalnum():
            y.append(i)

    # Replace the original list with filtered tokens and clear the temporary list
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Replace the original list with filtered tokens and clear the temporary list
    text = y[:]
    y.clear()

    # Apply stemming to reduce words to their root form
    for i in text:
        y.append(ps.stem(i))

    # Join the processed tokens into a single string
    return " ".join(y)

# Load the pre-trained TF-IDF vectorizer and classification model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Load the TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))  # Load the classification model

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Text area for user input
input_sms = st.text_area("Enter the message")  # Allow users to input the message

# Button for predicting spam or not
if st.button('Predict'):
    # Step 1: Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Step 2: Convert the preprocessed text into vector form using TF-IDF
    vector_input = tfidf.transform([transformed_sms])
    # Step 3: Predict the category (spam or not spam) using the model
    result = model.predict(vector_input)[0]
    # Step 4: Display the prediction result
    if result == 1:
        st.header("Spam")  # If the model predicts spam
    else:
        st.header("Not Spam")  # If the model predicts not spam
