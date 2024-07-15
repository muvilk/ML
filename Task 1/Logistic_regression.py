import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradient_descent(X, y, y_hat):
    # X - Input
    # y - true value
    # y_hat - predictions
    
    # m - number of training examples
    m = X.shape[0]
    
    # Gradient of loss with respect to weight
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    
    # Gradient of loss w.r.t bias
    db = (1 / m) * np.sum(y_hat - y)
    
    return dw, db

def normalize(X, epsilon=1e-8):
    # X - Input
    return (X - X.mean(axis=0)) / (X.std(axis=0) + epsilon)

def train(X, y, bs, epochs, lr):
    # X - Feature matrix
    # y - Labels
    # bs - Batch size
    # epochs - Number of epochs
    # lr - Learning rate
    
    n_samples, n_features = X.shape
    
    # Initializing weights and bias
    w = np.zeros((n_features, 1))
    b = 0
    
    # Reshaping y
    y = y.reshape(n_samples, 1)
    
    # Normalizing the inputs
    X = normalize(X)
    
    for epoch in range(epochs):
        for i in range(0, n_samples, bs):
            X_batch = X[i:i + bs]
            y_batch = y[i:i + bs]
            
            y_hat = sigmoid(np.dot(X_batch, w) + b)
            dw, db = gradient_descent(X_batch, y_batch, y_hat)
            
            w -= lr * dw
            b -= lr * db
    
    return w, b

def predict(X, w, b):
    X = normalize(X)
    y_hat = sigmoid(np.dot(X, w) + b)
    # Empty List to store predictions.
    pred = []
    # if y_hat > 0.5 round up to 1
    # if y_hat < 0.5 round to 0
    pred = [1 if i > 0.5 else 0 for i in y_hat]
    
    return np.array(pred)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load Data
data = pd.read_csv('Project/Data/train_tfidf_features.csv')
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Split the data
indices = np.arange(X_features.shape[0])
np.random.shuffle(indices)
split_idx = int(X_features.shape[0] * 0.8)
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

X_train_split = X_features[train_indices]
y_train_split = Y_label[train_indices]
X_val_split = X_features[val_indices]
y_val_split = Y_label[val_indices]

# Train the model
batch_size = 64
epochs = 1000
learning_rate = 0.01

w, b = train(X_train_split, y_train_split, batch_size, epochs, learning_rate)

# Predict
y_pred = predict(X_val_split, w, b)

# Accuracy
acc = accuracy(y_val_split, y_pred)
print(f'Validation Accuracy: {acc * 100}')
