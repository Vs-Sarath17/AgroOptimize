from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define a function to preprocess input data
def preprocess_input(data):
    # Extract numerical features
    numerical_features = [
        float(data['n']),
        float(data['p']),
        float(data['k']),
        float(data['pH']),
        float(data['rain']),
        float(data['temp'])
    ]
    return np.array(numerical_features).reshape(1, -1)

# Routes and other Flask app code...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    data = {
        'n': request.form['n'],
        'p': request.form['p'],
        'k': request.form['k'],
        'pH': request.form['pH'],
        'rain': request.form['rain'],
        'temp': request.form['temp']
    }

    # Preprocess input data
    processed_data = preprocess_input(data)

    # Make predictions
    prediction = model.predict(processed_data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
