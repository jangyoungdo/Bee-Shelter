import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, scaler, pca):
    # Standardize the data
    features = data.drop(columns=['label'])
    labels = data['label']
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    return features_pca, labels

def evaluate_model(model, features, labels):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return accuracy, precision, recall, f1

def main():
    # Load the test data
    filepath = 'data/test_data.csv'
    data = load_data(filepath)

    # Load the saved model, PCA, and scaler
    model = joblib.load('models/logistic_regression_model.pkl')
    pca = joblib.load('models/pca.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # Preprocess the test data
    features, labels = preprocess_data(data, scaler, pca)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, features, labels)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

if __name__ == "__main__":
    main()
