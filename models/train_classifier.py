import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import pickle


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Loads data from database and return X, Y and category_names.
    
    Arguments:
    database_filepath: filepath for source database
    
    Returns:
    X: Input variables
    Y: Target variables in dataframe
    category_names: names of the target variables
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)
    # split in X and Y
    X = df['message']
    Y = df[[col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]]
    category_names = Y.columns

    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenizes a given text (including cleaning, normalization and lemmatizing)

    Arguments:
    text: text string to be tokenized
    
    Returns:
    clean_tokens: list of tokenized words
    '''
    # remove special characters
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds a classifier using a Pipeline.
    
    Arguments:
    None
    
    Returns:
    cv: model after using GridSearch
    '''

    randomforest = RandomForestClassifier()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(randomforest))
    ])
    
    #reduced numer of parameters so that GridSearch does not take too long
    parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__max_features': (None, 5000),
            'clf__estimator__n_estimators': [20, 50],
            #'clf__estimator__min_samples_split': [2, 3],
            'clf__estimator__criterion': ['gini', 'entropy'],
            'clf__estimator__max_depth': [2, None],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluates a model based on the test predictions and prints the output.
    
    Arguments:
    model: model to be evaluated
    X_test: input variables for test prediction
    y_test: true output variables
    category_names: list of names of output variables
    
    Returns:
    None
    '''
    y_pred = model.predict(X_test)
    for i, c in enumerate(y_test.columns): 
        print("category: ", c) 
        print(classification_report(y_test.values[i], y_pred[i]))


def save_model(model, model_filepath):
    '''
    Saves a ML model to a given filepath a pickle file.
    
    Arguments:
    model: model to be saved
    model_filepath: target filepath of model pickel file
    
    Returns:
    None
    '''
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