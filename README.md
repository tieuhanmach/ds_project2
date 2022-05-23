# Disaster Response Pipeline Project
## **Motivation**
This project utilize data set containing genuine messages that were sent during disaster events and build machine learning pipeline to classify these occasions in order to send the messages to an suitable disaster help office. It includes a web app that applies the algorithm to an emergency worker's message as they receive it and show classification results in particular categories. The web app will also display visualizations of the information.

## **Instructions** :
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## **Libraries**
This project use built-in libraries for data cleaning ,text processing and data modeling:
`import pandas as pd`

`import numpy as np`

`import sqlite3`

`import re`

`import nltk`

`from nltk.corpus import stopwords`

`from nltk.stem.wordnet import WordNetLemmatizer`

`from nltk.tokenize import word_tokenize, sent_tokenize`

`from nltk import pos_tag`

`from sqlalchemy import create_engine`

`from sklearn.pipeline import Pipeline,FeatureUnion`

`from sklearn.metrics import confusion_matrix, classification_report`

`from sklearn.model_selection import train_test_split, GridSearchCV`

`from sklearn.multioutput import MultiOutputClassifier`

`from sklearn.ensemble import RandomForestClassifier`

`from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer`

`from sklearn.base import BaseEstimator, TransformerMixin`

`nltk.download('punkt')`

`nltk.download('stopwords')`

`nltk.download('wordnet')`

`nltk.download('averaged_perceptron_tagger')`
## **Table of content**
1. app
- template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
- run.py  # Flask file that runs app

2. data
- disaster_categories.csv  # data to process 
- disaster_messages.csv  # data to process
- process_data.py
- DisasterResponse.db   # database to save clean data to

3. models
- train_classifier.py
- classifier.pkl  # saved model 
