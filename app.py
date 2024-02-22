# app.py

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import re
import nltk
import joblib

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

app = Flask(__name__)
socketio = SocketIO(app)

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for binary cyberbullying detection
def binary_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = sgd_classifier.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
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
        return None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('text_from_extension')
def handle_text_from_extension(data):
    text = data.get('text', '')
    binary_result, offensive_words = binary_cyberbullying_detection(text)
    multi_class_result = multi_class_cyberbullying_detection(text)

    result_data = {
        'binary_result': binary_result,
        'offensive_words': offensive_words,
        'multi_class_result': multi_class_result
    }

    emit('result_to_extension', result_data)

# Expose an endpoint for your extension to send data
@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    binary_result, offensive_words = binary_cyberbullying_detection(text)
    multi_class_result = multi_class_cyberbullying_detection(text)

    result_data = {
        'binary_result': binary_result,
        'offensive_words': offensive_words,
        'multi_class_result': multi_class_result
    }

    return jsonify(result_data)

if __name__ == '__main__':
    socketio.run(app, debug=True)
