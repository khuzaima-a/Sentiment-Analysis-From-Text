from save_model import save_trained_model
from testing import perform_testing
from predict import predict_sentiment
import os.path

def sentiment_analysis():
    if(not os.path.exists('sentiment-analysis.pkl')): #Model is not trained & saved
        save_trained_model()
    perform_testing()
    predict_sentiment()

sentiment_analysis()
