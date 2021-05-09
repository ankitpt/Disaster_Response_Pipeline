import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
#from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys

sys.path.append("../models")

from storm_word_counter import StormWordCounter

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Msg_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #Finding count of each category and storing their names in a list
    cat_counts = df[df.columns[4:]].sum()
    cat_names = df[df.columns[4:]].sum().index

    #Finding co-occurrence matrix for each category
    coocc = df[df.columns[4:]].T.dot(df[df.columns[4:]])
    
    #Setting diagonal values to 0
    np.fill_diagonal(coocc.values, 0)

    #Finding top 20 most occurring pair of categories in message dataset
    top_N = 20
    
    #Considering all except first two categories ie related and request as they 
    #occur too often and are very generic
    ncoocc=coocc.values[2:,2:]
    
    #Since co-occurrence matrix is symmteric, we set upper triangular part of 
    #the matrix to 0  so that we do not consider a pair twice
    iu1 = np.triu_indices(34)
    ncoocc[iu1]=0
    
    #Extracting names of concerned categories
    cats_considered=df.columns[6:]
    
    #Finding IDs to of top 20 values in co-occurence matrix
    idx = np.argpartition(ncoocc, ncoocc.size - top_N, axis=None)[-top_N:]

    results = np.column_stack(np.unravel_index(idx, ncoocc.shape))


    
    freq=[]
    cats=[]
    for result in results:
        freq.append(ncoocc[result[0],result[1]])
        cats.append([cats_considered[result[0]],cats_considered[result[1]]])
    
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages under various categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[cat[0]+"+"+cat[1] for cat in cats],
                    y=freq
                )
            ],

            'layout': {
                'title': 'Top 20 co-occurring categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'type':'category',
                    'title': "Category pair",
                    'tickangle':30
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()