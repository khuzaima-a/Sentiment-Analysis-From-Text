import csv
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Will give name of next version of the file
# e.g: If current filename is test-v0.csv
# All modifications will be stored in file named test-v1.csv
def getName(filename):
    newfile = ''
    versionNo = filename[-5]
    newfile = filename[:-5:]
    versionNo = str(int(versionNo) + 1)
    newfile = newfile + versionNo + '.csv'
    return newfile

encodings = ['utf-8', 'latin-1', 'utf-16']

def read_csv_with_encoding(filename, encoding):
    with open(filename, 'r', encoding=encoding, errors='replace') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def get2Cols(filename,c1,c2):
    # Try reading the CSV file with different encodings

    newfile = getName(filename)
    if os.path.isfile('./' + newfile):
        return

    data = None
    for encoding in encodings:
        try:
            data = read_csv_with_encoding(filename, encoding)
            break
        except UnicodeDecodeError:
            continue

    # Select the desired columns
    selected_columns = [[row[c1], row[c2]] for row in data]  # Replace indices 0 and 1 with the desired column indices
    print("\033[92m",newfile + " created!!", "\033[0m")
    # Write the selected columns to a new CSV file
    with open(newfile, 'w', newline='', encoding='utf-8-sig') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(selected_columns)

#------------------------------------------------------------------------

def extractFeatures(filename):
    newfile = getName(filename)
    
    if os.path.isfile('./' + newfile):
        return
    
    nltk.download('stopwords')
    nltk.download('punkt')
    # Load the training data
    
    cleaned_messages = []
    selected_messages = []
    sentiments = []

    with open(filename, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message = row['text']
            sentiment = row['sentiment']

            # Removing special characters, punctuation, and numbers
            message = re.sub(r'[^\w\s]', '', message)
            message = re.sub(r'\d+', '', message)

            # Converting text to lowercase
            message = message.lower()

            # Tokenization
            tokens = word_tokenize(message)

            # Removing stop words
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token not in stop_words]

            cleaned_message = ' '.join(filtered_tokens)
            if(cleaned_message == ''):
                cleaned_messages.append(message)
            else:
                cleaned_messages.append(cleaned_message)
            selected_messages.append(message)
            sentiments.append(sentiment)

    # Create a new CSV file with three columns
    fieldnames = ['message', 'selected_messages', 'sentiment']

    with open(newfile, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(cleaned_messages)):
            writer.writerow({
                'message': selected_messages[i],
                'selected_messages': cleaned_messages[i],
                'sentiment': sentiments[i]
            })

    print(f"\033[93m{newfile} created!!\033[0m")

# -----------------------------------------------------------------------------

def clean_data():
    # Get 2 useful columns from training data
    get2Cols('train-v0.csv',1,3)
    
    # Get 2 useful columns from testing data
    get2Cols('test-v0.csv',1,2)

    # Now, applying operations on the data
    # Removing Punctuation
    # Removing Stop Words
    # Lowering the text
    # Tokenization

    # For training data
    extractFeatures('train-v1.csv')

    # For testing data
    extractFeatures('test-v1.csv')
