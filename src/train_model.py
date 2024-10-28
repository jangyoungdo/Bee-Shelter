import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Standardize the data
    scaler = StandardScaler()
    features = data.drop(columns=['Label'])
    labels = data['Label']
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels, scaler

def apply_pca(features, n_components=7):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca, pca

def pseudo_labeling(model, X_unlabeled, threshold=0.95):
    # Generate pseudo labels for unlabeled data based on a confidence threshold
    probabilities = model.predict_proba(X_unlabeled)
    high_confidence_mask = np.max(probabilities, axis=1) > threshold
    pseudo_labels = model.predict(X_unlabeled)[high_confidence_mask]
    return high_confidence_mask, pseudo_labels

def train_model(features, labels):
    model = LogisticRegression(solver='liblinear')
    model.fit(features, labels)
    return model

def main():
    # Load and preprocess the labeled data (100 hornet and 100 bee samples)
    filepath_bee = 'data/mfcc_honeybee.csv'
    filepath_hornet = 'data/mfcc_hornet.csv'
    df_mfcc_bee = load_data(filepath_bee)
    df_mfcc_hornet = load_data(filepath_hornet)

    n_samples = 100
    df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
    df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

    # Label assignment
    df_mfcc_bee_sampled['Label'] = 'B'
    df_mfcc_hornet_sampled['Label'] = 'H'

    # Combine bee and hornet data
    df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0).reset_index(drop=True)

    # PCA application
    X = df_combined.drop(columns=['Label'])
    y = df_combined['Label']
    features_pca, pca = apply_pca(X)
    y_encoded = pd.get_dummies(y).values.argmax(axis=1)

    # Split data into Train, Validation, Test
    X_train, X_temp, y_train, y_temp = train_test_split(features_pca, y_encoded, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train model on labeled data
    model = train_model(X_train, y_train)

    # Load unlabeled data and generate pseudo labels
    unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index).reset_index(drop=True)
    X_unlabeled = pca.transform(unlabeled_hornet)
    high_confidence_mask, pseudo_labels = pseudo_labeling(model, X_unlabeled)

    # If there are high-confidence pseudo-labeled samples, add them to the training data
    if np.any(high_confidence_mask):
        pseudo_labeled_data = unlabeled_hornet[high_confidence_mask].copy()
        pseudo_labeled_data['Label'] = pseudo_labels

        # Combine with original training data
        df_pseudo_combined = pd.concat([df_combined, pseudo_labeled_data], ignore_index=True).reset_index(drop=True)

        # Prepare data for re-training
        X_pseudo = df_pseudo_combined.drop(columns=['Label'])
        y_pseudo = pd.get_dummies(df_pseudo_combined['Label']).values.argmax(axis=1)
        features_pca_pseudo = pca.transform(X_pseudo)

        # Retrain the model with pseudo-labeled data
        X_train_pseudo, X_temp_pseudo, y_train_pseudo, y_temp_pseudo = train_test_split(features_pca_pseudo, y_pseudo, test_size=0.4, random_state=42)
        X_val_pseudo, X_test_pseudo, y_val_pseudo, y_test_pseudo = train_test_split(X_temp_pseudo, y_temp_pseudo, test_size=0.5, random_state=42)

        model = train_model(X_train_pseudo, y_train_pseudo)

        # Evaluate the model
        y_test_pseudo_pred = model.predict(X_test_pseudo)
        pseudo_test_accuracy = accuracy_score(y_test_pseudo, y_test_pseudo_pred)
        print(f'Test Accuracy with Pseudo Labeling: {pseudo_test_accuracy:.2f}')

    # Save the model, PCA, and scaler
    joblib.dump(model, 'models/logistic_regression_model.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    joblib.dump(StandardScaler(), 'models/scaler.pkl')

if __name__ == "__main__":
    main()
