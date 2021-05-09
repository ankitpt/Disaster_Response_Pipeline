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

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Msg_table',engine)
    X = df.message.values
    Y = df[df.columns[4:]].values.astype('int64')
    return X,Y,df.columns[4:]

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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


def build_model():
        
    pipeline3 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('storm_count',  StormWordCounter())
        ])),
        ('scale', StandardScaler(with_mean=False)),
        ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
    ])

    parameters = {

                'features__text_pipeline__vect__max_df': [0.5],
                    'features__text_pipeline__vect__max_features': [None],
                        'clf__estimator__n_estimators': [100],}
        #'cl                ator__min_samples_split': [ 3, 4],}

    cv_new = GridSearchCV(pipeline3, param_grid=parameters,cv=3)        
    return cv_new

def evaluate_model(model, X_test, Y_test, category_names):
        Y_pred = model.predict(X_test)
        f1_scores_nalg=np.zeros((36,1))
        for i,col in enumerate(category_names):
            print(f"{col} performance metrics")
            print(classification_report(Y_test[:,i], Y_pred[:,i]))
            f1_scores_nalg[i,]=f1_score(Y_test[:,i], Y_pred[:,i])
        print("##########################################################")
        print(f"Average f1-score for this model is {np.mean(f1_scores_nalg)}")

def save_model(model, model_filepath):
    import pickle
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()