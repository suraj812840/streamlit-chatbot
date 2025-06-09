import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

# Load required files
model = load_model("chatbot_model1.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = [intent['tag'] for intent in intents['intents']]

# Preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [{"intent": classes[i], "probability": str(r)} for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results if results else [{"intent": "noanswer", "probability": "0"}]

def getResponse(ints):
    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Talk to the AI assistant below!")

user_input = st.text_input("You:")

if user_input:
    ints = predict_class(user_input)
    response = getResponse(ints)
    st.text_area("Bot:", value=response, height=100)
