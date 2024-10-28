import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def load_data(bee_path, hornet_path):
    df_mfcc_bee = pd.read_csv(bee_path)
    df_mfcc_hornet = pd.read_csv(hornet_path)
    return df_mfcc_bee, df_mfcc_hornet

def initialize_threshold_dicts(confidence_thresholds):
    return (
        {threshold: [] for threshold in confidence_thresholds},
        {threshold: [] for threshold in confidence_thresholds},
        {threshold: [] for threshold in confidence_thresholds},
    )

def split_data(df_mfcc_bee, df_mfcc_hornet, n_samples):
    df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
    df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)
    df_mfcc_bee_sampled['Label'] = 'B'
    df_mfcc_hornet_sampled['Label'] = 'H'
    df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0)
    return df_combined, df_mfcc_bee_sampled, df_mfcc_hornet_sampled

def prepare_train_val_test_sets(df_combined):
    X = df_combined.drop(columns=['Label'])
    y = pd.get_dummies(df_combined['Label']).values.argmax(axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model

def pseudo_labeling(logreg, unlabeled_data, confidence_thresholds):
    X_unlabeled = unlabeled_data
    probs = logreg.predict_proba(X_unlabeled)
    pseudo_labeling_data_counts = {threshold: [] for threshold in confidence_thresholds}
    total_sample_counts = {threshold: [] for threshold in confidence_thresholds}
    remaining_unlabeled_counts = {threshold: [] for threshold in confidence_thresholds}

    for threshold in confidence_thresholds:
        high_confidence_mask = np.max(probs, axis=1) > threshold
        pseudo_labeled_data = unlabeled_data[high_confidence_mask].copy()
        pseudo_labeled_count = len(pseudo_labeled_data)
        pseudo_labeling_data_counts[threshold].append(pseudo_labeled_count)
        total_sample_count = len(X_train) + pseudo_labeled_count
        total_sample_counts[threshold].append(total_sample_count)
        remaining_unlabeled_counts[threshold].append(len(unlabeled_data) - pseudo_labeled_count)

    return pseudo_labeling_data_counts, total_sample_counts, remaining_unlabeled_counts

def plot_results(n_samples_list, total_sample_counts_by_threshold, remaining_unlabeled_data_counts_by_threshold, confidence_thresholds):
    plt.figure(figsize=(12, 8))
    for threshold in confidence_thresholds:
        plt.plot(n_samples_list, total_sample_counts_by_threshold[threshold], marker='x', linestyle='-', label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Total (Guided + Pseudo-labeled) Sample Count')
    plt.title('Total Sample Count vs. Number of Samples at Different Confidence Thresholds')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    for threshold in confidence_thresholds:
        plt.plot(n_samples_list, remaining_unlabeled_data_counts_by_threshold[threshold], marker='o', linestyle='-', label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Remaining Unlabeled Data Count')
    plt.title('Remaining Unlabeled Data Count vs. Number of Samples at Different Confidence Thresholds')
    plt.grid(True)
    plt.legend()
    plt.show()

bee_path = '/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_honeybee.csv'
hornet_path = '/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_hornet.csv'
df_mfcc_bee, df_mfcc_hornet = load_data(bee_path, hornet_path)

n_samples_list = list(range(100, 6100, 100))
confidence_thresholds = np.arange(0.5, 1.0, 0.05)
(
    pseudo_labeling_data_counts_by_threshold,
    total_sample_counts_by_threshold,
    remaining_unlabeled_data_counts_by_threshold,
) = initialize_threshold_dicts(confidence_thresholds)

for n_samples in n_samples_list:
    df_combined, df_mfcc_bee_sampled, df_mfcc_hornet_sampled = split_data(df_mfcc_bee, df_mfcc_hornet, n_samples)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_sets(df_combined)
    logreg = train_logistic_regression(X_train, y_train)

    unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index)
    unlabeled_bee = df_mfcc_bee.drop(df_mfcc_bee_sampled.index)
    unlabeled_data = pd.concat([unlabeled_hornet, unlabeled_bee], axis=0)

    pseudo_counts, total_counts, remaining_counts = pseudo_labeling(logreg, unlabeled_data, confidence_thresholds)

    for threshold in confidence_thresholds:
        pseudo_labeling_data_counts_by_threshold[threshold].append(pseudo_counts[threshold][-1])
        total_sample_counts_by_threshold[threshold].append(total_counts[threshold][-1])
        remaining_unlabeled_data_counts_by_threshold[threshold].append(remaining_counts[threshold][-1])

plot_results(n_samples_list, total_sample_counts_by_threshold, remaining_unlabeled_data_counts_by_threshold, confidence_thresholds)
