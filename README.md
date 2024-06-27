# Consumer-Segmentation-ML-DL-WebApp


<img width="1434" alt="Screenshot 2024-06-27 at 21 27 19" src="https://github.com/s-mehakkk/Consumer-Segmentation-ML-DL-WebApp/assets/75841992/8ba80a07-0d8f-4774-a6ba-28b0b4aea5aa">

## Description
This project involves analyzing a consumer dataset using various machine learning and deep learning techniques. The primary objectives include performing exploratory data analysis (EDA), applying classification and clustering algorithms, and deploying the models in a simple web interface built with Flask. The models are exported using joblib and hosted on the server for real-time predictions.

## Exploratory Data Analysis
The EDA was performed to understand the dataset and extract meaningful insights. The analysis includes visualizations and statistical summaries.

## Machine Learning Models
We applied the following classification algorithms:
1. Decision Tree
2. Random Forest
3. Naive Bayes

The models were trained and evaluated to classify customers based on their tenures. The best-performing model was selected based on accuracy.

## Deep Learning Models
Two deep learning techniques were applied to segment the customers:
1. Technique 1
2. Technique 2

The models were optimized to yield better clustering results.

## Web Interface
A simple web interface was created using Flask. The interface allows users to input customer details and get the predicted customer segment. It also displays pie and bar charts to show the distribution of customer segments.

## Usage
1. To start the Flask server:
    ```zsh
    python app.py
    ```
2. Open your web browser and go to:
    ```text
    http://127.0.0.1:5000/
    ```
3. Enter the required input details to get the customer segment prediction.

Source code:
Python - Assignment_Round.ipynb
Server - app.py


