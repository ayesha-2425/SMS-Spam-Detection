import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tk = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))

st.title("📩 SMS Spam Detection Model")
st.write("This is an NLP application to classify messages as spam or not spam.")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):

    #1. preprocess
    transformed_sms = transform_text(input_sms)
    #2.vectorizer
    vector_input = tk.transform([transformed_sms])
    #3.Predict
    result = model.predict(vector_input)[0]
    #4.Display
    if result == 1:
        st.header("🚨 Spam Message!")
    else:
        st.header("✅ Not Spam")import streamlit as st
import pickle
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ✅ **Manually keep important spam words**
spam_keywords = {"free", "win", "claim", "prize", "urgent", "credit", 
                 "cash", "award", "guaranteed", "offer", "bonus", "money", 
                 "lottery", "congratulations", "selected", "winner", "urgent", "txt", "text"}
stop_words -= spam_keywords  

# ✅ **Preprocessing function**
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    # Keep only alphanumeric words + important symbols (£, $)
    tokens = [word for word in tokens if word.isalnum() or word in ["£", "$"]]

    # Remove stopwords but keep spam-related words
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    return " ".join(stemmed_tokens)


# Load vectorizer & model
vectorizer_path = "vectorizer.pkl"
model_path = "model.pkl"

if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    st.error("🚨 Required model files not found! Ensure 'vectorizer.pkl' and 'model.pkl' are present.")
    st.stop()

with open(vectorizer_path, "rb") as file:
    tk = pickle.load(file)

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ✅ **Streamlit UI**
st.title("📩 SMS Spam Detection Model")
st.write("*This is an NLP application to classify messages as spam or not spam.*")

input_sms = st.text_area("Enter the SMS")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # 1. Preprocess input
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize input
        vector_input = tk.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.header("🚨 Spam Message!")
        else:
            st.header("✅ Not Spam")
