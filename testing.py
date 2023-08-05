import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

def perform_testing():
    # Load the trained model
    trained_model = pickle.load(open('sentiment-analysis.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # Load the testing data
    test_data = pd.read_csv('test.csv')

    # Split the text messages into individual words
    test_messages = test_data['selected_messages'].str.split()

    # Create a new DataFrame with word-level messages
    test_word_data = pd.DataFrame({'selected_messages': test_messages})

    # Transform the testing data using the fitted vectorizer
    # vectorizer.transform() when we have already 
    # fitted the vectorizer on the training data
    test_features = vectorizer.transform(test_word_data['selected_messages'].apply(' '.join))
    
    # Predict the sentiment labels for the testing data using the loaded model
    test_predictions = trained_model.predict(test_features)

    # Print the accuracy score for the predictions
    accuracy = accuracy_score(test_data['sentiment'], test_predictions)
    accuracy = round(accuracy * 100)
    print("\nAccuracy of the model is: ", accuracy, "%")