import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler


class StormWordCounter(BaseEstimator, TransformerMixin):
    # Adding 'activate' parameter to activate the transformer or not:
    def __init__(self, activate = True):
        self.activate = activate

        # Defining fit method:
    def fit(self, X, y = None):
            return self

        # Defining transform method:
    def transform(self, X):
            '''
            It recieves an array of messages and counts the number of characters
            for each message.
            Input:
            X: array of text messages
            Output:
            elec_words_arr: array with the number of storm words for each
            message.
            '''
            # If activate parameter is set to True:
            if self.activate:
                st_words_count = list()
                st_list = ['rain','raining','storm','cyclone']#['shelter', 'home','house', 'housing', 'tent']
                # Counting shelter words:
                for text in X:
                    # Creating empty list:
                    st_words = 0
                    tokens = word_tokenize(text.lower())
                    for word in tokens:
                        if word in st_list:
                            st_words += 1
                    st_words_count.append(st_words)
                            # Transforming list into array:
                st_words_arr = np.array(st_words_count)
                st_words_arr = st_words_arr.reshape((len(st_words_arr), 1))
                return st_words_arr

                        # If activate parameter is set to False:
            else:
                            pass
