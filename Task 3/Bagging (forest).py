import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV

df = pd.read_csv('data/train_tfidf_features.csv')
k = 5

# Separate features and labels
X = df.drop(columns=['id', 'label'])
y = df['label']

model = RandomForestClassifier(n_estimators=400, random_state=42, criterion="entropy", min_samples_split=80, max_features='sqrt', max_samples=0.6, 
                               class_weight='balanced_subsample', max_depth=150, min_samples_leaf=2)

if False:  # Change this to True to activate hyperparameter optimization
    param_grid = {
        'n_estimators': [350,360,370,380,390]
    }
    
    print("Doing hyperparameter optimization")
    halving_grid_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        factor=3,  # The factor by which the number of candidates is reduced at each iteration
        scoring='f1_macro',
        n_jobs=-1,
        cv=5,
        verbose=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    halving_grid_search.fit(X_train, y_train)
    print("Best parameters found: ", halving_grid_search.best_params_)

    # Collecting results
    results = pd.DataFrame(halving_grid_search.cv_results_)

    # Best model evaluation
    best_model = halving_grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    print(f"F1 Score: {f1}")

if True:
    print("Doing Cross Validation")
    scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
    print(f"Cross-Validation F1 Scores for {k} folds: {scores}")
    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())

if True:
    test = pd.read_csv("data/test_tfidf_features.csv")
    X_test = test.drop(['id'], axis=1).values
    model.fit(X, y)
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame({'id': test['id'], 'label': y_pred})

    # Display the first few rows of the predictions
    print(predictions_df.head())

    # Save predictions to a CSV file
    predictions_df.to_csv('Optimal_forest', index=False)

if False: #creating graphs for each hyperparameter
    min_samples_split_range = range(2, 100, 1)

    # Store the mean and standard deviation of the F1 Macro scores for each n_estimators
    mean_scores = []
    std_scores = []

    for min_samples_split in min_samples_split_range:
        print(f"Evaluating model with min_samples_split={min_samples_split}")
        model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy",
                                       min_samples_split=min_samples_split, max_features='sqrt', max_samples=0.7,
                                       class_weight='balanced_subsample', max_depth=80, min_samples_leaf=1)
        scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        print(f"Mean F1 Macro Score: {scores.mean()} | Std Dev: {scores.std()}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(min_samples_split_range, mean_scores, yerr=std_scores, fmt='-o')
    plt.xlabel('min_samples_split')
    plt.ylabel('Mean F1 Macro Score')
    plt.title('F1 Macro Score vs. min_samples_split')
    plt.grid(True)
    plt.savefig('min_samples_split_vs_f1_macro.png')
    plt.close()

    max_depth_range = range(1, 200, 1)

    # Store the mean and standard deviation of the F1 Macro scores for each n_estimators
    mean_scores = []
    std_scores = []

    for max_depth in max_depth_range:
        print(f"Evaluating model with max_depth={max_depth}")
        model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy",
                                       min_samples_split=7, max_features='sqrt', max_samples=0.7,
                                       class_weight='balanced_subsample', max_depth=max_depth, min_samples_leaf=1)
        scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        print(f"Mean F1 Macro Score: {scores.mean()} | Std Dev: {scores.std()}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(max_depth_range, mean_scores, yerr=std_scores, fmt='-o')
    plt.xlabel('max_depth')
    plt.ylabel('Mean F1 Macro Score')
    plt.title('F1 Macro Score vs. max_depth')
    plt.grid(True)
    plt.savefig('max_depth_vs_f1_macro.png')
    plt.close()

    min_samples_leaf_range = range(1, 100, 1)

    # Store the mean and standard deviation of the F1 Macro scores for each n_estimators
    mean_scores = []
    std_scores = []

    for min_samples_leaf in min_samples_leaf_range:
        print(f"Evaluating model with min_samples_leaf={min_samples_leaf}")
        model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy",
                                       min_samples_split=7, max_features='sqrt', max_samples=0.7,
                                       class_weight='balanced_subsample', max_depth=80, min_samples_leaf=min_samples_leaf)
        scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        print(f"Mean F1 Macro Score: {scores.mean()} | Std Dev: {scores.std()}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(min_samples_leaf_range, mean_scores, yerr=std_scores, fmt='-o')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Mean F1 Macro Score')
    plt.title('F1 Macro Score vs. min_samples_leaf')
    plt.grid(True)
    plt.savefig('min_samples_leaf_vs_f1_macro.png')
    plt.close()

    max_samples_range = np.arange(0.01, 1, 0.01)

    # Store the mean and standard deviation of the F1 Macro scores for each n_estimators
    mean_scores = []
    std_scores = []

    for max_samples in max_samples_range:
        print(f"Evaluating model with max_samples={max_samples}")
        model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy",
                                       min_samples_split=7, max_features='sqrt', max_samples=max_samples,
                                       class_weight='balanced_subsample', max_depth=80, min_samples_leaf=1)
        scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        print(f"Mean F1 Macro Score: {scores.mean()} | Std Dev: {scores.std()}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(max_samples_range, mean_scores, yerr=std_scores, fmt='-o')
    plt.xlabel('max_samples')
    plt.ylabel('Mean F1 Macro Score')
    plt.title('F1 Macro Score vs. max_samples')
    plt.grid(True)
    plt.savefig('max_samples_vs_f1_macro.png')
    plt.close()

    n_estimators_range = range(10, 510, 10)

    # Store the mean and standard deviation of the F1 Macro scores for each n_estimators
    mean_scores = []
    std_scores = []

    for n_estimators in n_estimators_range:
        print(f"Evaluating model with n_estimators={n_estimators}")
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, criterion="entropy",
                                       min_samples_split=7, max_features='sqrt', max_samples=0.7,
                                       class_weight='balanced_subsample', max_depth=80, min_samples_leaf=1)
        scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        print(f"Mean F1 Macro Score: {scores.mean()} | Std Dev: {scores.std()}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_estimators_range, mean_scores, yerr=std_scores, fmt='-o')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean F1 Macro Score')
    plt.title('F1 Macro Score vs. n_estimators')
    plt.grid(True)
    plt.savefig('n_estimators_vs_f1_macro.png')
    plt.close()