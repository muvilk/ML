import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load training data
data = pd.read_csv("C:/Users/muvil/Documents/Muvil/SUTD/Term 5/50.007 Machine Learning/Project/Data/train_tfidf_features.csv")
X_features = data.drop(['label', 'id'], axis=1).values
Y_label = data['label'].values

# Load test data
test = pd.read_csv("C:/Users/muvil/Documents/Muvil/SUTD/Term 5/50.007 Machine Learning/Project/Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1).values

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_features, Y_label, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = model.predict(X_val)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)

# Print the accuracy
print("Validation Accuracy:", accuracy)

# Make predictions on the test data
y_final = model.predict(X_test)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_final})
predictions_df.to_csv('LogReg_Prediction_SKLearn.csv', index=False)
