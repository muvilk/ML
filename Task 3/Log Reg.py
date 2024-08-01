import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Define your train, predict, and accuracy functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradient_descent(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db

def train(X, y, bs, epochs, lr, X_val, y_val):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    b = 0
    y = y.reshape(n_samples, 1)
    
    training_loss = []
    validation_f1 = []
    validation_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, n_samples, bs):
            X_batch = X[i:i + bs]
            y_batch = y[i:i + bs]
            y_hat = sigmoid(np.dot(X_batch, w) + b)
            dw, db = gradient_descent(X_batch, y_batch, y_hat)
            w -= lr * dw
            b -= lr * db

            epoch_loss += loss(y_batch, y_hat)
        
        epoch_loss /= (n_samples // bs)
        training_loss.append(epoch_loss)

        # Calculate validation F1 score and loss
        y_hat_val = sigmoid(np.dot(X_val, w) + b)
        y_pred_val = (y_hat_val > 0.5).astype(int)
        f1 = f1_score(y_val, y_pred_val, average='macro')
        validation_f1.append(f1)
        val_loss = loss(y_val, y_hat_val)
        validation_loss.append(val_loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation F1: {f1:.4f}, Validation Loss: {val_loss:.4f}')

    return w, b, training_loss, validation_f1, validation_loss

def predict(X, w, b):
    y_hat = sigmoid(np.dot(X, w) + b)
    pred = [1 if i > 0.5 else 0 for i in y_hat]
    return np.array(pred)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def optimize_bagging(X_features, Y_label):
    n_bags = 5  # Number of bags
    batch_size_options = [32, 64, 128, 256]
    learning_rate_options = [0.001, 0.01, 0.1, 0.2]
    epochs = 500

    models = []
    accuracies = []
    f1_scores = []
    indices = np.arange(X_features.shape[0])
    all_training_losses = []
    all_validation_f1s = []
    all_validation_losses = []

    for i in range(n_bags):
        np.random.shuffle(indices)
        split_idx = int(X_features.shape[0] * 0.85)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        X_train_split = X_features[train_indices]
        y_train_split = Y_label[train_indices]
        X_val_split = X_features[val_indices]
        y_val_split = Y_label[val_indices]

        best_f1 = 0
        best_params = {}
        for lr in learning_rate_options:
            for bs in batch_size_options:
                w, b, training_loss, validation_f1, validation_loss = train(X_train_split, y_train_split, bs, epochs, lr, X_val_split, y_val_split)
                y_pred = predict(X_val_split, w, b)
                f1 = f1_score(y_val_split, y_pred, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'learning_rate': lr, 'batch_size': bs}

                # Store training loss, validation F1 score, and validation loss for plotting
                all_training_losses.append((training_loss, f'LR: {lr}, BS: {bs}'))
                all_validation_f1s.append((validation_f1, f'LR: {lr}, BS: {bs}'))
                all_validation_losses.append((validation_loss, f'LR: {lr}, BS: {bs}'))

        best_lr = best_params['learning_rate']
        best_bs = best_params['batch_size']
        w, b, _, _, _ = train(X_train_split, y_train_split, best_bs, epochs, best_lr, X_val_split, y_val_split)
        models.append((w, b))
        f1_scores.append(best_f1)
        print(f'Model {i + 1} Trained with LR: {best_lr}, BS: {best_bs}, F1 Score: {best_f1:.4f}')

    # Plot training loss and validation loss for each hyperparameter combination
    for loss, label in all_training_losses:
        plt.plot(loss, label=f'Train {label}')
    for loss, label in all_validation_losses:
        plt.plot(loss, label=f'Val {label}')
    plt.title('Training and Validation Loss for Different Hyperparameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot validation F1 score for each hyperparameter combination
    for f1, label in all_validation_f1s:
        plt.plot(f1, label=label)
    plt.title('Validation F1 Score for Different Hyperparameters')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

    # Plot F1 Score vs lr
    for lr in learning_rate_options:
        subset = f1_df[f1_df['Learning Rate'] == lr]
        plt.plot(subset['Batch Size'], subset['F1 Score'], marker='o', label=f'LR: {lr}')
    plt.title('F1 Score vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

    # Plot F1 Score vs batch size
    for bs in batch_size_options:
        subset = f1_df[f1_df['Batch Size'] == bs]
        plt.plot(subset['Learning Rate'], subset['F1 Score'], marker='o', label=f'BS: {bs}')
    plt.title('F1 Score vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

    best_idx = np.argmax(f1_df['F1 Score'])
    best_params = f1_df.iloc[best_idx]
    best_lr = best_params['Learning Rate']
    best_bs = best_params['Batch Size']
    best_f1 = best_params['F1 Score']
    
    best_model_idx = np.argmax([f1 for _, _, f1 in f1_results])
    best_model = models[best_model_idx]

    print(f'Best Parameters: Learning Rate = {best_lr}, Batch Size = {best_bs}, F1 Score = {best_f1:.4f}')
    return best_model, best_f1

# Load the training data (replace with your actual data loading logic)
data = pd.read_csv("Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Train the optimized bagging model
best_model, best_f1_score = optimize_bagging(X_features, Y_label)
best_w, best_b = best_model

# Load the test data and first set of predictions
test = pd.read_csv("Data/test_tfidf_features.csv")

# Predict on the test data using the best model
X_test = test.drop(columns=['id']).values
y_pred_best = predict(X_test, best_w, best_b)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({
    'id': test['id'],
    'label': y_pred_best
})

# Display the first few rows of the final predictions
print(predictions_df.head())

predictions_df.to_csv('Optimal_LogReg_predictions.csv', index=False)

# Completed Print
print("Final predictions saved to Optimal_LogReg_predictions.csv")

#Model 3 Trained with LR: 0.1, BS: 32, F1 Score: 0.7008