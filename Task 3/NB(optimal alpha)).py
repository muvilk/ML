import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt

# Step 1: Load the Training Data
train_file_path = 'Data/train_tfidf_features.csv'  # Path to the training CSV file
train_data = pd.read_csv(train_file_path)

# Step 2: Extract Features and Labels
X_train = train_data.drop(columns=['id', 'label'])  # Drop 'id' and 'label' columns to get features
Y_train = train_data['label']  # Extract the 'label' column as target labels

# Step 3: Set Up a Detailed Hyperparameter Grid for Alpha
alpha_values = np.arange(0.2, 0.3, 0.005)  # More detailed range of alpha values

# Step 4: Perform K-Fold Cross-Validation for Each Alpha
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

# Step 5: Plot the Results to Find the Optimal Alpha
plt.figure(figsize=(10, 6))
plt.plot(results_df['Alpha'], results_df['F1 Score'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('F1 Score (Macro)')
plt.title('Alpha vs. F1 Score (Macro)')
plt.grid(True)
plt.show()

# Step 6: Find the Exact Optimal Alpha Value
optimal_alpha = results_df.loc[results_df['F1 Score'].idxmax()]['Alpha']
optimal_f1_score = results_df.loc[results_df['F1 Score'].idxmax()]['F1 Score']

print(f"Optimal alpha value: {optimal_alpha}")
print(f"F1 Score at optimal alpha: {optimal_f1_score}")


