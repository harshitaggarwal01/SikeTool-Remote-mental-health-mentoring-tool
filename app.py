from __future__ import unicode_literals
from flask import Flask, render_template, request

from Summarizers.spacy_summarization import text_summarizer
import pickle
# text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
import spacy
from tensorflow.keras.models import load_model
import re
from nltk import word_tokenize
import pandas as pd
import numpy as np


num_classes = 5
max_seq_len = 500
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']


classifier = pickle.load(open('models/model.pkl','rb'))
cv = pickle.load(open('models/cv.pkl','rb'))
predictor_cnn = load_model("models/LSTM_with_word2vec.h5")
tokenizer=Tokenizer()
nlp = spacy.load('en')
app = Flask(__name__)
output = []


def reemovNestings(l):
    for i in l:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)


def clean_text(data):
    # Removing Punctuation Marks, unecessary symbols and everything:
    data = re.sub(r'\[[0-9]*\]', ' ', str(data))
    data = re.sub(r'[^\w\s]', '', data)
    data = re.sub(r'\s+', ' ', data)
    data = data.lower()
    data = re.sub(r'\d', ' ', data)
    data = re.sub(r'\s+', ' ', data)
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # tekenization using nltk
    data = word_tokenize(data)

    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in data if not w in stop_words]
    data = ' '.join(filtered_sentence)

    return data

def pred_mod(text):
    message = clean_text(text)

    # messages=[' '.join(x) for x in message]
    mess = ' '.join(message)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(mess)
    seq = tokenizer.texts_to_sequences(mess)
    reemovNestings(seq)
    pad = pad_sequences([output], maxlen=max_seq_len)
    pred = predictor_cnn.predict(pad)

    return class_names[np.argmax(pred)]

# Reading Time
def readingTime(mytext):
    total_words = len([token.text for token in nlp(mytext)])
    estimatedTime = total_words / 200.0
    return estimatedTime


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        preds=pred_mod(rawtext)
        data=[rawtext]
        vect = cv.transform(data).toarray()
        my_predict = classifier.predict(vect)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
    return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,
                           final_reading_time=final_reading_time, summary_reading_time=summary_reading_time, predictions=preds,depress=my_predict)


@app.route('/about')
def about():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
