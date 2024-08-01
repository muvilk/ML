import pandas as pd
import numpy as np

test = pd.read_csv("Data/test_tfidf_features.csv")

# Read the first set of predictions
y_pred1 = pd.read_csv("Data/Optimal_LogReg.csv")
y_pred1 = y_pred1['label'].values

# Read the second set of predictions
y_pred2 = pd.read_csv("Data/Optimal_gbdt.csv")
y_pred2 = y_pred2['label'].values

# Read the third set of predictions
y_pred3 = pd.read_csv("Data/Optimal_forest.csv")
y_pred3 = y_pred3['label'].values

# Read the fourth set of predictions
y_pred4 = pd.read_csv("Data/Optimal_NB.csv")
y_pred4 = y_pred4['label'].values

# Read the fifth set of predictions
y_pred5 = pd.read_csv("Data/Optimal_svm.csv")
y_pred5 = y_pred5['label'].values

# Combine predictions into a DataFrame for easier manipulation
predictions_df = pd.DataFrame({
    'id': test['id'],
    'pred1': y_pred1,
    'pred2': y_pred2,
    'pred3': y_pred3,
    'pred4': y_pred4,
    'pred5': y_pred5
})

# Perform majority voting
predictions_df['label'] = (predictions_df[['pred1', 'pred2', 'pred3', 'pred4', 'pred5']].sum(axis=1) > 1).astype(int)

#Display the first few rows of the final predictions
print(predictions_df.head())

# Save the final predictions to a CSV file
final_df = predictions_df[['id', 'label']]
final_df.to_csv('BestModel_Predictions.csv', index=False)