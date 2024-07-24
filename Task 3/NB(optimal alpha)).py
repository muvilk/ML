import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer

# Step 1: Load the Training Data
train_file_path = 'Data/train_tfidf_features.csv'  # Path to the training CSV file
train_data = pd.read_csv(train_file_path)

# Step 2: Extract Features and Labels
X_train = train_data.drop(columns=['id', 'label'])  # Drop 'id' and 'label' columns to get features
Y_train = train_data['label']  # Extract the 'label' column as target labels

# Step 3: Set Up the Hyperparameter Grid for Alpha
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

# Step 4: Perform K-Fold Cross-Validation for Each Alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average='macro')

results = {}

for alpha in param_grid['alpha']:
    model = MultinomialNB(alpha=alpha)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring=scorer)
    results[alpha] = np.mean(cv_scores)
    print(f"Alpha: {alpha}, F1 Score: {np.mean(cv_scores)}")

# Step 5: Display the Best Alpha Value Based on Cross-Validation Performance
best_alpha = max(results, key=results.get)
print(f"Best alpha value based on cross-validation: {best_alpha}")
print(f"Best cross-validation F1 score: {results[best_alpha]}")

# Optionally, display detailed results
results_df = pd.DataFrame(list(results.items()), columns=['Alpha', 'F1 Score'])
print("Detailed Results:")
print(results_df)



