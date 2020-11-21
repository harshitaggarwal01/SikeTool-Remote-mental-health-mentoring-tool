from __future__ import unicode_literals
from flask import Flask, render_template, request
from Summarizers.spacy_summarization import text_summarizer
import time
import spacy
import re
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

predictor=ktrain.load_predictor('models/bert_model')
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)
output = []

classifier = pickle.load(open('models/model.pkl','rb'))
cv = pickle.load(open('models/cv.pkl','rb'))

# Reading Time
def readingTime(mytext):
    total_words = len(mytext)
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
        preds=predictor.predict(final_summary)
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
    app.run(debug=True, use_reloader=False)
