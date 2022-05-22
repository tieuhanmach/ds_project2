import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """Load data from proved database

    Args:  
    database_filepath: text data
    Returns:
    X: feature variables
    y: target variables
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_cleaned',engine)
    X=df.message.values
    y=df[df.columns[4:]].values
    categories_name=list(df.columns[4:])
    return X, y,categories_name


def tokenize(text):
    """Clean text: normalize case, remove punctuation,
    tokenize text, lemmatize and remove stop words

    Args:  
    text: text data
    Returns:
    Cleaned text
    """
    # Normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]',' ', text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")]
    return tokens


def build_model():
    """Build model to classify categories from messages
    
    Returns:
    cv: tunned model ready to fit and predict
    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ]
    )

    # specify parameters for grid search
    parameters = {
             'clf__estimator__n_estimators': [50, 100]}

    # create grid search object
    cv = GridSearchCV(pipeline,param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate model has been built 

    Args:  
    model: tunned model ready to fit and predict
    X_test: feature variable in test dataset
    y_test: target categories in test dataset
    category_names: List of categories name
    Returns:
    classification_report with f1 score, precision and recall for each output category
    """
    result_report={}
    for ind,cat in enumerate(category_names):
        y_test_cat = Y_test[ind]
        y_pred = model.predict(X_test)
        y_pred_cat=y_pred[ind]
        report=classification_report(y_test_cat, y_pred_cat, labels=[0,1])
        result_report[cat]=report
    return result_report


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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