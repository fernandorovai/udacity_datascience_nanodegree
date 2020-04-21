import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

# Opens database, extracts data to a dataframe and return X and Y
def load_data(database_filepath):
    """
    Loads data from database
    
    Input:
        database_filepath: String, path to database to connect
    Returns:
        X: Dataframe, X data for training and testing
        Y: Dataframe, labels for X data, also for training and testing
        category_names: Names for categories in Y
    """
        
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("msgs", engine.connect())
    X = df["message"]
    Y = df.drop(labels=["message", "original", "genre"], axis=1)
    return X,Y, Y.columns

def tokenize(text):
    """
    Transform to lower case, remove symbols and numbers, tokenize and remove stopwords
    
    Input:
        text: String, tweet from a supposed natural disaster event
    Returns:
        clean_tokens: List of strings, tokenized and clean words
    """
    
    textVal = text.lower() #transform all to lowercase
    textVal = re.sub(r"[^a-zA-Z]", " ", textVal) #replace symbols / accents
    tokens = word_tokenize(textVal) #tokenize words
    tokens = [w for w in tokens if w not in stopwords.words("english")] #remove stopwords (at, on, and, not, ...)
    return tokens

def build_model():
     """
    Creates machine learning pipeline for learning
    
    Input:
        None:
    Returns:
        pipeline: Pipeline with tokenization, tfidf embedding extraction and classifier
    """
    # stablish the pipeline: Tokenize using our custom function, apply TF-IDF for getting embeddings and train the classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LogisticRegression(random_state = 5)), n_jobs=-1))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model in X and Y data
    
    Go through each category name and evaluate using X and Y
    
    Input:
        model: Model trained on X_train and Y_train
        X_test: Dataframe, validation data for model
        Y_test: Dataframe, actual labels for the test data in X
        category_names: List of strings, categories to be evaluated
    Returns:
        None: Prints out report to terminal
    """
        
    Y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):    
        y_true = list(Y_test.values[:, i])
        y_pred = list(Y_pred[:, i])
        target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)]
        print(classification_report(y_true, y_pred, target_names=target_names))

def save_model(model, model_filepath):
    """
    Saves model as a pickle file to model_filepath
    
    Input:
        model: pipeline/model, to be pickled for later use
        model_filepath: String, filepath where model will be saved
    Returns:
        None: Pickle file will be created at model_path
    """
        
    joblib.dump(model, model_filepath)


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

        # save tokens
        joblib.dump(model.named_steps['vect'].get_feature_names(), 'tokens.pkl')
          
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