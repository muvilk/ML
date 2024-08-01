import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
import joblib

# Loading the training data
data = pd.read_csv('Data/train_tfidf_features.csv')
X = data.drop(columns=['id', 'label'])
Y = data['label']


'''
MultinomialNB has 2 main parameters (alpha and fit_prior)
alpha is the smoothing parameter, preventing the model from assigning 0 probability to unseen words
adds alpha to count when calculating conditional probability. (default =1)
fit_prior is a boolean that sets whether the model will calculate prior probabilities from the training set. (default = True)
Also important to note that SciKit uses log likelihood when calculating 
'''
#Set Up a Detailed Hyperparameter Grid for Alpha
alpha_values = np.arange(0.01, 1, 0.01)  # More detailed range of alpha values

#Perform K-Fold Cross-Validation for Each Alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average='macro')

results = []

for alpha in alpha_values:
    model = MultinomialNB(alpha=alpha)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring=scorer)
    results.append((alpha, np.mean(cv_scores)))
    print(f"Alpha: {alpha}, F1 Score: {np.mean(cv_scores)}")

# Convert results to a DataFrame for easier plotting
results_df = pd.DataFrame(results, columns=['Alpha', 'F1 Score'])

#Plot the Results to Find the Optimal Alpha
plt.figure(figsize=(10, 6))
plt.plot(results_df['Alpha'], results_df['F1 Score'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('F1 Score (Macro)')
plt.title('Alpha vs. F1 Score (Macro)')
plt.grid(True)
plt.show()

#Find the Exact Optimal Alpha Value
optimal_alpha = results_df.loc[results_df['F1 Score'].idxmax()]['Alpha']
optimal_f1_score = results_df.loc[results_df['F1 Score'].idxmax()]['F1 Score']

print(f"Optimal alpha value: {optimal_alpha}")
print(f"F1 Score at optimal alpha: {optimal_f1_score}")


# Initialize the model
model = MultinomialNB(alpha = 0.26)

# Load the test data
test = pd.read_csv("Data/test_tfidf_features.csv")
X_test = test.drop(['id'], axis=1)

# Fit the model on the entire training data
model.fit(X, Y)

# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Create a DataFrame with predictions
predictions_df = pd.DataFrame({'id': test['id'], 'label': y_test_pred})

# Save predictions to a CSV file
#predictions_df.to_csv('Optimal_NB.csv', index=False) #uncomment to get file, otherwise file is found in /task3_predictions


