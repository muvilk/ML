import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
import joblib

# Step 1: Load the Data
file_path = 'Data/train_tfidf_features.csv'  # Path to the CSV file
data = pd.read_csv(file_path)

# Step 2: Extract Features and Labels
X = data.drop(columns=['id', 'label'])  # Drop 'id' and 'label' columns to get features
Y = data['label']  # Extract the 'label' column as target labels

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Initialize and Train the Model
model = MultinomialNB()  # Create an instance of the model
model.fit(X_train, Y_train)

# Step 5: Evaluate the Model
Y_pred = model.predict(X_test)
f1 = f1_score(Y_test, Y_pred, average='macro')
report = classification_report(Y_test, Y_pred)

print(f"Macro F1 score: {f1}")
print(f"Classification Report:\n{report}")

