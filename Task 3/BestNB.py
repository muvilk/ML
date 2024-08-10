import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.metrics import f1_score, classification_report

# Loading the training data
data = pd.read_csv('Data/train_tfidf_features.csv')
X = data.drop(columns=['id', 'label'])
Y = data['label']

# Initialize the models with default alpha values
models = {
    'MultinomialNB': MultinomialNB(),  # Default alpha = 1.0
    'GaussianNB': GaussianNB(),        # No alpha parameter
    'BernoulliNB': BernoulliNB(),      # Default alpha = 1.0
    'ComplementNB': ComplementNB()     # Default alpha = 1.0
}

# Perform K-Fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=100)

# Store the results
results = {}

for model_name, model in models.items():
    cross_validation_scores = cross_val_score(model, X, Y, cv=kf, scoring='f1_macro')
    average_score = cross_validation_scores.mean()
    results[model_name] = average_score
    
    # Predict with cross-validation and print classification report
    Y_pred = cross_val_predict(model, X, Y, cv=kf)
    print(f"Classification Report for {model_name}:")
    print(classification_report(Y, Y_pred))

# Print the average Macro F1 score for each model
for model_name, score in results.items():
    print(f"Average Macro F1 score for {model_name}: {score}")