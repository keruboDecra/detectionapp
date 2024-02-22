
import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image
import sys  # Import the sys module
# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.base import clone
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model_pipeline = joblib.load('sgd_classifier_model.joblib')
new_model_pipeline = None

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the logo image
logo = Image.open('logo.png')
import streamlit as st
import tornado.web
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop

import streamlit as st
import tornado.web
import tornado.ioloop
from tornado.wsgi import WSGIContainer


# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)



# Function for binary cyberbullying detection
def binary_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        st.error(f"Error in binary_cyberbullying_detection: {e}")
        return None, None

# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction
        decision_function_values = sgd_classifier.decision_function([preprocessed_text])[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(decision_function_values)

        # Get the predicted class label using the label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_class_label, decision_function_values
    except Exception as e:
        st.error(f"Error in multi_class_cyberbullying_detection: {e}")
        return None

# Create a new Tornado handler for the highlighted text endpoint
class HighlightedTextHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            selected_text = data.get('selected_text', '')

            if selected_text:
                # Perform classification using your existing functions
                binary_result, _ = binary_cyberbullying_detection(selected_text)
                multi_class_result = multi_class_cyberbullying_detection(selected_text)

                # Return the classification results as JSON
                self.write({
                    'binary_result': binary_result,
                    'multi_class_result': multi_class_result[0]
                })
            else:
                self.write({'error': 'No selected text received'})

        except Exception as e:
            st.error(f"Error in HighlightedTextHandler: {e}")
            self.write({'error': 'Internal server error'})

# Create a new Tornado application and add the handler
tornado_app = tornado.web.Application([
    ('/highlighted_text', HighlightedTextHandler),
])

# Set up the Tornado server
tornado_server = tornado.httpserver.HTTPServer(tornado_app)
tornado_server.listen(8888)  # You can choose a different port

# ... (existing code)

# Check if the app is being used by the Chrome extension
if 'selected_text' in st.session_state:
    receive_highlighted_text()

# ... (existing code)

# Start the Tornado server along with the Streamlit app
st.title('Cyberbullying Detection App')
st.write("Streamlit content goes here")

# Start the Tornado server in a separate thread
tornado_thread = threading.Thread(target=tornado.ioloop.IOLoop.current().start)
tornado_thread.start()

# Start the Streamlit app
st.run()




