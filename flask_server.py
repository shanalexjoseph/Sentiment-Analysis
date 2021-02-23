# import the necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.models import load_model

import time
import html
from nltk.tokenize import WordPunctTokenizer
import re
import pickle
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import flask
import io
import sys

from flask import render_template

tok = WordPunctTokenizer()

mention1 = r'@[A-Za-z0-9]+'
mention2 = r'https?://[^ ]+'
combined_pat = r'|'.join((mention1, mention2))
www_pat = r'www.[^ ]+\.[^ ]+'
negations_dic = {"don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not", "wasn't":"was not", "weren't":"were not", "isn't":"is not", "aren't":"are not", 
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "dislike":"do not like"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
combined_pattern = re.compile(combined_pat)
www_pattern = re.compile(www_pat)
letters_pattern = re.compile("[^a-zA-Z]")

# Sentence cleaning
def clean_sentence(lower, higher, texts):
  
  results = []
  start_time = time.time()
  
  for i in range(lower, higher):
    
    if (i - lower + 1) % 100000 == 0:
      end_time = time.time() - start_time
      print(i - lower + 1, "Sentences cleaned , time in seconds:", end_time)
      start_time = time.time()
    text = texts[i]
    html_unescaped = html.unescape(text)
    try:
        bom_removed = html_unescaped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = html_unescaped
    stripped = combined_pattern.sub('', bom_removed)
    stripped = www_pattern.sub('', stripped)
    lower_case = stripped.lower()
    words = [negations_dic[x] if x in negations_dic else x for x in lower_case.split(' ')]
    neg_handled = " ".join(words).strip()
    letters_only = letters_pattern.sub(" ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    results.append(" ".join(words).strip())
    
  return results

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tfidf = None

@app.route("/predict", methods=["GET"])
def predict():
    text = flask.request.args.get("text")
    prediction_string=""
    prediction_value=""
    if text != None:
        new_sentence = [text]
        clean_Sentence = clean_sentence(0, len(new_sentence), new_sentence)
        print(clean_Sentence, sys.stderr)
        pred_sentence = tfidf.transform(clean_Sentence).toarray()
        pred = model.predict(pred_sentence, verbose=1)
        prediction_string = "Negative"
        prediction_value = pred[0][0]
        if prediction_value >= 0.5:
            prediction_string = "positive"
    return render_template("prediction.html", prediction_string=prediction_string, prediction_value=prediction_value)
                                                                                                               
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # loading the tfidf
    tfidf = pickle.load(open("tfidf_result_3.pkl", "rb" ) )
    # loading the model
    model = load_model('nn-model_3.h5')
    app.run(use_reloader=False)
