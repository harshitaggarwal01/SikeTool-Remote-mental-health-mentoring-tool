from __future__ import unicode_literals
from flask import Flask, render_template, request
from spacy_summarization import text_summarizer
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
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from keras.models import load_model
import copy
import sys


#### MONGODB STUFF
from pymongo import MongoClient
from datetime import date

client = MongoClient("mongodb+srv://harshit:harshit@virtualdiary1.a0xns.mongodb.net/diary?retryWrites=true&w=majority")
db = client.get_database('diary')

def create_new_user(username, password):
    collection_name = username + ":" + password
    db.create_collection(collection_name)

def create_new_entry(username, password, text, metric):
    collection_name = username + ":" + password
    
    records=db[collection_name]
    
    today = date.today()

    d1 = today.strftime("%d/%m/%Y")
    
    entry={
        'date' : d1,
        'text' : text,
        'metric' : metric
    }
    
    records.insert_one(entry)

def view_all_entries(username, password):
    collection_name = username + ":" + password
    
    records=db[collection_name]
    
    all_entries = list(records.find())
    
    return all_entries

def view_one_entry(username, password, date):
    collection_name = username + ":" + password
    
    records=db[collection_name]
    
    one_entry = records.find_one({'date': date})
    
    return one_entry

def delete_one_entry(username, password, date):
    collection_name = username + ":" + password
    
    records=db[collection_name]
    
    records.delete_one({'date': date})

def get_mongo_det(dict_data):
  dict_date=dict_data['date']
  dict_text=dict_data['text']
  dict_metric=dict_data['metric']
  print("Date = ",dict_date)
  print("Text = ",dict_text)
  print("Metric = ",dict_metric)

  return dict_date,dict_text,dict_metric

def get_dep_prob(text):
  data=[text]
  vect = cv.transform(data).toarray()
  my_predict_prob = classifier.predict_proba(vect)
  prob=my_predict_prob[0][1]
  return prob
#####

'''
def get_dep_prob(text):
  data=[text]
  vect = cv.transform(data).toarray()
  my_predict_prob = classifier.predict_proba(vect)
  prob=my_predict_prob[0][1]
  return prob

def get_emotion_prob(text):
  emotion_prob = predictor.predict(text, return_proba=True)
  #print(emotion_prob)
  depression_indicator = (emotion_prob[1] + emotion_prob[2] + emotion_prob[3]) - (emotion_prob[0]+emotion_prob[4])
  #print(depression_indicator)
  return depression_indicator

def cam_predict():
        model=load_model('D:\CLICK HERE\python\octahacks\Tm_Okthcks_pvte_Mod-main\Virtual Diary Mod\models\model_weights.h5')
        face_cascade = cv2.CascadeClassifier('D:\CLICK HERE\python\octahacks\Tm_Okthcks_pvte_Mod-main\Virtual Diary Mod\models\haarcascade_frontalface_default.xml')
        cap=cv2.VideoCapture(0)
        predict_lst = []
        while True:
            ret, frame = cap.read()
            img = copy.deepcopy(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                fc = gray[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
                print(pred)
                text_idx = np.argmax(pred)
                predict_lst.append(pred)
            break
        depression_indicator_list = []
        for i in predict_lst:
            for j in i:
                depression_indicator = (j[0] + j[1] + j[2] + j[5]) - (j[3]+j[4]+j[6])
                depression_indicator_list.append(depression_indicator)
        return depression_indicator_list

def getDepressionLevel(text):
    face_emotion_prob = cam_predict()
    face_prob = np.max(face_emotion_prob)
    #some function to get the submitted entry text from mongoDB
    binary_prob = get_dep_prob(text)
    emotion_prob = get_emotion_prob(text)
    metric = (binary_prob+emotion_prob+face_prob)/3
    return metric
'''
predictor=ktrain.load_predictor('models/bert_model')
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)
output = []

classifier = pickle.load(open('models/model.pkl','rb'))
cv = pickle.load(open('models/cv.pkl','rb'))
predict_lst = []
# Reading Time
def readingTime(mytext):
    total_words = len(mytext)
    estimatedTime = total_words / 200.0
    return estimatedTime

# Camera_Prediction
"""
Calling this function after loading the home page and ending it with analysis page will give the max emmtion during the period
"""
@app.route("/res", methods=['GET', 'POST'])
def cam_predict():    
    if request.method=='GET':
        model=load_model('models/model_weights.h5')
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        cap=cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            img = copy.deepcopy(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                fc = gray[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
                text_idx = np.argmax(pred)
                text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                if text_idx == 0:
                    text = text_list[0]
                elif text_idx == 1:
                    text = text_list[1]
                elif text_idx == 2:
                    text = text_list[2]
                elif text_idx == 3:
                    text = text_list[3]
                elif text_idx == 4:
                    text = text_list[4]
                elif text_idx == 5:
                    text = text_list[5]
                elif text_idx == 6:
                    text = text_list[6]
                predict_lst.append(text)
                time.sleep(5)
    cap.release()
    
    start = time.time()
    if request.method == 'POST':
        if predict_lst.count("Happy")>5 or predict_lst.count("Sad")>5:
            if predict_lst.count("Happy") > predict_lst.count("Sad"):
                total_predict = "Happy"
            else:
                total_predict = "Sad"
        else:
            total_predict = max(predict_lst, key=predict_lst.count)

        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        data=[rawtext]
        vect = cv.transform(data).toarray()
        my_pred=classifier.predict(vect)
        my_predict_prob = classifier.predict_proba(vect)
        prob = my_predict_prob[0][1]
        preds = predictor.predict(final_summary)
        emotion_prob = predictor.predict(final_summary, return_proba=True)
        depression_indicator = ((emotion_prob[1] + emotion_prob[2] + emotion_prob[3]) - (emotion_prob[0]+emotion_prob[4]) + prob)/2
        depression_indicator = int(depression_indicator * 100)
        metric = depression_indicator

        create_new_entry(username="harsh", password="123", text=rawtext, metric=metric)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
        return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,tp=total_predict,metric=metric,
                            final_reading_time=final_reading_time, summary_reading_time=summary_reading_time, predictions=preds, depress= my_pred)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        if predict_lst.count("Happy")>5 or predict_lst.count("Sad")>5:
            if predict_lst.count("Happy") > predict_lst.count("Sad"):
                total_predict = "Happy"
            else:
                total_predict = "Sad"
        else:
            total_predict = max(predict_lst, key=predict_lst.count)

        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        data=[rawtext]
        vect = cv.transform(data).toarray()
        my_predict_prob = classifier.predict_proba(vect)
        my_pred=classifier.predict(vect)
        prob = my_predict_prob[0][1]
        # prob=get_dep_prob(rawtext)
        preds = predictor.predict(final_summary)
        emotion_prob = predictor.predict(final_summary, return_proba=True)
        depression_indicator = ((emotion_prob[1] + emotion_prob[2] + emotion_prob[3]) - (emotion_prob[0]+emotion_prob[4]) + prob)/2
        depression_indicator = int(depression_indicator * 100)
        metric = depression_indicator

        create_new_entry(username="harsh", password="123", text=rawtext, metric=metric)
        # data=[rawtext]
        # vect = cv.transform(rawtext).toarray()
        # my_predict = classifier.predict(vect)
        #metric = getDepressionLevel(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
        # return render_template('index.html', ctext=rawtext, final_time=final_time,
        #                    final_reading_time=final_reading_time, depress=metric)
    return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,tp=total_predict,metric=metric,
                            final_reading_time=final_reading_time, summary_reading_time=summary_reading_time, predictions=preds,depress=my_pred)

@app.route('/about')
def about():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
