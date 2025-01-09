from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Credit Card Approval Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([data['features']])  # Convert input to NumPy array
        prediction = model.predict(features)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
