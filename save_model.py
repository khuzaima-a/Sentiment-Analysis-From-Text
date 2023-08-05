from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

def save_trained_model():
    # Load the training data (Dataframe)
    train_data = pd.read_csv('train.csv')

    # Split the text messages into individual words
    train_messages = train_data['selected_messages'].str.split()

    # Create a new DataFrame with words and sentiment
    # Adding sentiment column to the DataFrame
    train_word_data = pd.DataFrame({'selected_messages': train_messages, 'sentiment': train_data['sentiment']})
    
    # Initialize the CountVectorizer with n-grams
    # This will be accepting unigrams, bigrams and trigrams
    # and saving them in form of integers (count)
    vectorizer = CountVectorizer(ngram_range=(1, 2))

    # Fit the vectorizer to the training data
    # "apply(' '.join)" because CountVectorizer expects input as a collection of strings
    train_features = vectorizer.fit_transform(train_word_data['selected_messages'].apply(' '.join))

    # Saving the vectorizer in pkl file
    with open('vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
        print("\033[96mVectorizer saved!!\033[0m")

    # Train the Logistic Regression classifier
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(train_features, train_word_data['sentiment'])

    # Saving the model in pkl file
    with open('sentiment-analysis.pkl', 'wb') as file:
        pickle.dump(logistic_regression, file)
        print("\033[96mModel saved!!\033[0m")