from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load combined MFCC dataset
df_combined = pd.read_csv('mfcc_features.csv')

# Scale data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_combined.drop(columns=['Label']))

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df_combined['Cluster'] = kmeans.fit_predict(df_scaled)

# Encode labels
label_encoder = LabelEncoder()
df_combined['True_Label'] = label_encoder.fit_transform(df_combined['Label'])

# Calculate confusion matrix and adjusted accuracy
conf_matrix = confusion_matrix(df_combined['True_Label'], df_combined['Cluster'])
accuracy = max(conf_matrix[0, 0] + conf_matrix[1, 1], conf_matrix[0, 1] + conf_matrix[1, 0]) / len(df_combined)
print("Confusion Matrix:\n", conf_matrix)
print(f"Clustering Accuracy: {accuracy * 100:.2f}%")

# Visualize clustering result using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Plot K-means clustering result
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df_combined['Cluster'], cmap='viridis')
plt.title('K-means Clustering of MFCCs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Plot true labels
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df_combined['True_Label'], cmap='coolwarm')
plt.title('True Labels of MFCCs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='True Label')
plt.show()


