import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

with open("dectree_model.pkl","rb") as f:
    model = pickle.load(f)

st.title("Tweet Toxic/Non-toxic Predictor")
st.write("Bollu Uday Yadav")

tweet = st.text_input("Please enter your tweet")

import nltk
import re
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def Clean(text):
  regex = "[^A-Za-z\s]"
  text = re.sub(regex," ",text)
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]
  return " ".join(lemmatized_tokens)

if st.button("Predict Class"):
    cleaned_text = Clean(tweet)

    result = model.predict([cleaned_text])[0]

    if result == 0:
       st.success("Predicted Class : Non-Toxic")
    else:
       st.success("Predicted Class : Toxic")
    