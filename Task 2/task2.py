import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#reading data
df = pd.read_csv('data/train_tfidf_features.csv')

# Separate features and labels
X = df.drop(columns=['id', 'label'])
y = df['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2000)  # Reduce to 2000 dimensions, adjust as needed
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Macro F1 Score: {macro_f1:.2f}')

print('now training with 100% train data...')

print("2000 components...")
# Load training data
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=2000)  # Reduce to 2000 dimensions
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled) #apply same PCA transformation to test data

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_pca, Y_label)

# Make predictions
y_pred = knn.predict(X_test_pca)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_pred})
predictions_df.to_csv('PCA2000_Prediction.csv', index=False)

print("1000 components...")
# Load training data
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=1000)  # Reduce to 1000 dimensions
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled) #apply same PCA transformation to test data

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_pca, Y_label)

# Make predictions
y_pred = knn.predict(X_test_pca)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_pred})
predictions_df.to_csv('PCA1000_Prediction.csv', index=False)

print("500 components...")
# Load training data
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=500)  # Reduce to 500 dimensions
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled) #apply same PCA transformation to test data

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_pca, Y_label)

# Make predictions
y_pred = knn.predict(X_test_pca)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_pred})
predictions_df.to_csv('PCA500_Prediction.csv', index=False)

print("100 components...")
# Load training data
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=100)  # Reduce to 100 dimensions
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled) #apply same PCA transformation to test data

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_pca, Y_label)

# Make predictions
y_pred = knn.predict(X_test_pca)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_pred})
predictions_df.to_csv('PCA100_Prediction.csv', index=False)