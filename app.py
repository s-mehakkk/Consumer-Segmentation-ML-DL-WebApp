from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.cluster import DBSCAN

app = Flask(__name__)

# Load the models
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
rf_classifier = joblib.load('rf_classifier.pkl')

def predict_segment(input_data):
    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Print the scaled data for debugging
    print("Scaled Input Data:", input_data_scaled)

    # Predict if the input is core or noise using DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    try:
        dbscan_cluster = dbscan.fit_predict(input_data_scaled)
    except Exception as e:
        dbscan_cluster = [-1, -1]

    print("db cluster ", dbscan_cluster[0])

    if dbscan_cluster[0] == -1:
        # If noise, classify using the Random Forest classifier
        prediction = rf_classifier.predict(input_data_scaled)
        segment_type = 'Noise'
    else:
        # If core, classify using the KMeans model
        prediction = kmeans.predict(input_data_scaled)
        segment_type = 'Core'
    
    return segment_type, prediction[0]

@app.route('/')
def home():
    segment_counts = {'Core': 5468, 'Noise': 3482} 
    return render_template('index.html', segment_counts=segment_counts)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])

    # Convert data to appropriate types
    input_data = input_data.astype(float)

    # Predict the segment
    segment_type, predicted_segment = predict_segment(input_data)

    # Return the result as JSON
    return jsonify({
        'segment_type': segment_type,
        'predicted_segment': int(predicted_segment)  # Ensure the segment is int for JSON serialization
    })

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
