import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def predict_sentiment():
    while True:
        # User input for prediction
        user_text = input("\nEnter a text message (enter 'exit' to terminate): ")

        if user_text.lower() == 'exit':
            break

        # Any character that is not a word or whitespace is removed
        message = re.sub(r'[^\w\s]', '', user_text) 

        # Removing digits
        message = re.sub(r'\d+', '', message)

        # Converting text to lowercase
        message = message.lower()

        # Split the text messages into individual words
        message_words = word_tokenize(message)

        # Removing stop words
        # Stopwords are words that do not add much meaning to a sentence
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in message_words if word not in stop_words]

        if(filtered_words == []):
            filtered_words = message_words

        # Load the vectorizer
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

        # Transform the user words using the fitted vectorizer
        # Sparse matrix is returned
        # Giving index of the words present in the user input and their count in the user input
        user_feature = vectorizer.transform([' '.join(filtered_words)])

        # Load the trained model
        model = pickle.load(open('sentiment-analysis.pkl', 'rb'))

        # Predict the sentiment label for the user input
        user_prediction = model.predict(user_feature)[0]
        print("Predicted sentiment:", user_prediction)
