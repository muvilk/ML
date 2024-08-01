import numpy as np
import pandas as pd

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function for logistic regression
def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Gradient descent function to compute gradients
def gradient_descent(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db

# Training function for logistic regression using mini-batch gradient descent
def train(X, y, bs, epochs, lr, tol=1e-4):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))  # Initialize weights
    b = 0  # Initialize bias
    y = y.reshape(n_samples, 1)
    
    for epoch in range(epochs):
        # Shuffle the dataset
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Mini-batch gradient descent
        for i in range(0, n_samples, bs):
            X_batch = X_shuffled[i:i + bs]
            y_batch = y_shuffled[i:i + bs]
            
            # Compute predictions
            y_hat = sigmoid(np.dot(X_batch, w) + b)
            # Compute gradients
            dw, db = gradient_descent(X_batch, y_batch, y_hat)
            
            # Update weights and bias
            w -= lr * dw
            b -= lr * db

        # Early Stopping check
        if epoch % 100 == 0:
            y_hat_full = sigmoid(np.dot(X, w) + b)
            current_loss = loss(y, y_hat_full)
            print(f'Epoch {epoch}, Loss: {current_loss:.4f}')

            if epoch > 10 and abs(previous_loss - current_loss) < tol:
                print(f"Early stopping at epoch {epoch}")
                break
            previous_loss = current_loss
    
    return w, b

# Prediction function for logistic regression
def predict(X, w, b):
    y_hat = sigmoid(np.dot(X, w) + b)
    pred = [1 if i > 0.5 else 0 for i in y_hat]
    return np.array(pred)

# Accuracy calculation function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load training data
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values

# Split the data into training and validation sets
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

# Predict on the validation set
y_pred = predict(X_val_split, w, b)

# Calculate validation accuracy
acc = accuracy(y_val_split, y_pred)
print(f'Validation Accuracy: {acc * 100:.2f}%')

# Predict on the test set
y_final = predict(X_test, w, b)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_final})
predictions_df.to_csv('LogRed_Prediction.csv', index=False)
