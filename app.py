import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# # Check versions (optional)
# print("Flask version:", version("flask"))
# print("Joblib version:", version("joblib"))

# Load the model
model = joblib.load('random_forest_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive data in JSON format
    features = np.array(data['features']).reshape(1, -1)  # Format input for model

    # Predict with the model
    prediction = model.predict(features)

    # Send the prediction as JSON
    return jsonify({'recommendation': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
