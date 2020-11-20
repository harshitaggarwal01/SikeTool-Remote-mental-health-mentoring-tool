from __future__ import unicode_literals
from flask import Flask, render_template, request

from Summarizers.spacy_summarization import text_summarizer

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

predictor_cnn = load_model("models/cnn_w2v.h5")
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
    # remove hashtags and @usernames
    data = str(data)
    data = re.sub(r"(#[\d\w\.]+)", '', str(data))
    data = re.sub(r"(@[\d\w\.]+)", '', str(data))

    # tekenization using nltk
    data = word_tokenize(str(data))

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
        preds=pred_mod(final_summary)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
    return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,
                           final_reading_time=final_reading_time, summary_reading_time=summary_reading_time, predictions=preds)


@app.route('/about')
def about():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
