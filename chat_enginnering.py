import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# import data 
data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

# Preprocess Data 
cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

cust.reset_index(drop = True, inplace=True)
sales.reset_index(drop = True, inplace = True)

df = pd.DataFrame()
df['Questions'] = cust[1]
df['Answer'] = sales[1]

# Data Cleaning and Preprocessing 
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)

df['tokenized Questions'] = df['Questions'].apply(preprocess_text)

corpus = df['tokenized Questions'].to_list()

tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus)

def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return df['Answer'].iloc[most_similar_index]

import random
chatbot_greeting = [
    "Hello there, welcome to the company's Bot. pls ejoy your usage",
    "Hi user, this bot is here to give answers to your requests",
    "Hi hi, How you may I help you today",
    "Hello user, pleased to have you here today, how may I help",  
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']

print(f'\t\t\t\t\tWelcome To Orpheus ChatBot\n\n')
while True:
    user_q = input('You: ', f'You are on to Samsung Call Center: \n')
    if user_q in user_greeting:
        print(random.choice(chatbot_greeting))
    elif user_q in exit_word:
        print(f'Thank you for your usage. Bye\n')
        break
    else:
        responses = bot_response(user_q)
        print(f'ChatBot:  {responses}')
