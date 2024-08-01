import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
import joblib

# Step 1: Load the Training and Test Data
train_file_path = 'Data/train_tfidf_features.csv'  # Path to the training CSV file
test_file_path = 'Data/test_tfidf_features.csv'  # Path to the test CSV file

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Step 2: Extract Features and Labels
X_train = train_data.drop(columns=['id', 'label'])  # Drop 'id' and 'label' columns to get features
Y_train = train_data['label']  # Extract the 'label' column as target labels

X_test = test_data.drop(columns=['id'])  # Drop 'id' columns to get features

# Step 3: Initialize and Train the Model with alpha=0.26
model = MultinomialNB(alpha=0.26)
print("training")
model.fit(X_train, Y_train)

# Step 4: Make Predictions on the Test Data
print("NB predicting")
Y_pred = model.predict(X_test)

predictions_df = pd.DataFrame({'id': test_data['id'], 'label': Y_pred})
# Save the DataFrame to a CSV file
predictions_df.to_csv('Optimal_NB.csv', index=False)


