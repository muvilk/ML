import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values  # Convert to NumPy array
y = data['label'].values  # Convert to NumPy array

# List of random seeds
random_seeds = [42, 52, 62]

# List of boosting algorithms and weak learners
boosting_algorithms = {
    'LightGBM GBDT': LGBMClassifier(boosting_type='gbdt'),
    'LightGBM DART': LGBMClassifier(boosting_type='dart'),
    'LightGBM GOSS': LGBMClassifier(boosting_type='goss')
}

weak_learners = {
    'Decision Stump': DecisionTreeClassifier(max_depth=1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gaussian Naive Bayes': GaussianNB()
}

# Initialize dictionaries to store F1 scores and times for each algorithm/learner
boosting_results = {name: {'f1_scores': [], 'times': []} for name in boosting_algorithms}
weak_learner_results = {name: {'f1_scores': [], 'times': []} for name in weak_learners}

def evaluate_boosting(name, model, seed):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Initialize the model
    model.set_params(random_state=seed)
    
    # Measure the start time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Measure the end time
    end_time = time.time()
    
    # Calculate the time taken
    elapsed_time = end_time - start_time
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the macro F1 score on the test set
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Return the F1 score and time taken
    return name, seed, f1, elapsed_time

def evaluate_learner(name, base_estimator, seed):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Initialize the AdaBoost classifier
    adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, learning_rate=1.0, random_state=seed)
    
    # Measure the start time
    start_time = time.time()
    
    # Train the classifier
    adaboost.fit(X_train, y_train)
    
    # Measure the end time
    end_time = time.time()
    
    # Calculate the time taken
    elapsed_time = end_time - start_time
    
    # Predict on the test set
    y_pred = adaboost.predict(X_test)
    
    # Calculate the macro F1 score on the test set
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Return the F1 score and time taken
    return name, seed, f1, elapsed_time

# Use Parallel and delayed to parallelize the evaluation
results_boosting = Parallel(n_jobs=-1)(delayed(evaluate_boosting)(name, model, seed)
                                       for name, model in boosting_algorithms.items()
                                       for seed in random_seeds)

results_weak_learners = Parallel(n_jobs=-1)(delayed(evaluate_learner)(name, base_estimator, seed)
                                            for name, base_estimator in weak_learners.items()
                                            for seed in random_seeds)

# Process results for boosting algorithms
for name, seed, f1, elapsed_time in results_boosting:
    boosting_results[name]['f1_scores'].append(f1)
    boosting_results[name]['times'].append(elapsed_time)

# Process results for weak learners
for name, seed, f1, elapsed_time in results_weak_learners:
    weak_learner_results[name]['f1_scores'].append(f1)
    weak_learner_results[name]['times'].append(elapsed_time)

# Calculate the mean and standard deviation of F1 scores and times for each algorithm/learner
boosting_means_f1 = {name: np.mean(boosting_results[name]['f1_scores']) for name in boosting_algorithms}
boosting_stds_f1 = {name: np.std(boosting_results[name]['f1_scores']) for name in boosting_algorithms}
boosting_means_time = {name: np.mean(boosting_results[name]['times']) for name in boosting_algorithms}
boosting_stds_time = {name: np.std(boosting_results[name]['times']) for name in boosting_algorithms}

learner_means_f1 = {name: np.mean(weak_learner_results[name]['f1_scores']) for name in weak_learners}
learner_stds_f1 = {name: np.std(weak_learner_results[name]['f1_scores']) for name in weak_learners}
learner_means_time = {name: np.mean(weak_learner_results[name]['times']) for name in weak_learners}
learner_stds_time = {name: np.std(weak_learner_results[name]['times']) for name in weak_learners}

# Combine the results for plotting
all_names = list(boosting_means_f1.keys()) + list(learner_means_f1.keys())
all_means_f1 = list(boosting_means_f1.values()) + list(learner_means_f1.values())
all_stds_f1 = list(boosting_stds_f1.values()) + list(learner_stds_f1.values())
all_means_time = list(boosting_means_time.values()) + list(learner_means_time.values())
all_stds_time = list(boosting_stds_time.values()) + list(learner_stds_time.values())

# Create a dual dot plot
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot F1 scores
color = 'tab:blue'
ax1.set_xlabel('Algorithm / Weak Learner')
ax1.set_ylabel('Mean Macro F1 Score', color=color)
ax1.errorbar(all_names, all_means_f1, yerr=all_stds_f1, fmt='o', capsize=5, color=color, label='Mean F1 Score')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis to plot times
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Mean Time Taken (seconds)', color=color)
ax2.errorbar(all_names, all_means_time, yerr=all_stds_time, fmt='o', capsize=5, color=color, label='Mean Time Taken')
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and grid
plt.title('Mean Macro F1 Score and Mean Time Taken vs. Algorithm with Error Bars')
ax1.grid(True)
fig.tight_layout()
plt.xticks(rotation=45)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

# Print the mean and standard deviation of F1 scores and times for each algorithm/learner
for name in all_names:
    if name in boosting_means_f1:
        print(f"Boosting Algorithm: {name}, Mean F1 Score: {boosting_means_f1[name]:.2f}, Std: {boosting_stds_f1[name]:.2f}")
        print(f"Boosting Algorithm: {name}, Mean Time Taken: {boosting_means_time[name]:.2f} seconds, Std: {boosting_stds_time[name]:.2f}")
    else:
        print(f"Weak Learner: {name}, Mean F1 Score: {learner_means_f1[name]:.2f}, Std: {learner_stds_f1[name]:.2f}")
        print(f"Weak Learner: {name}, Mean Time Taken: {learner_means_time[name]:.2f} seconds, Std: {learner_stds_time[name]:.2f}")

        import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values
y = data['label'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# List of hyperparameters and their ranges
hyperparameters = {
    'num_leaves': [15, 31, 63, 127, 255],
    'n_estimators': [50, 100, 200, 400, 800],
    'max_depth': [-1, 3, 5, 7, 9, 12],
    'min_child_samples': [10, 20, 50, 100, 150],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 5, 10, 20],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 2]
}

# Initialize a dictionary to store results
results = {param: {'train_f1_scores': [], 'test_f1_scores': []} for param in hyperparameters}

def evaluate_hyperparameter(param_name, param_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    for value in param_values:
        print(f"Evaluating {param_name} with value {value}...")
        train_f1_scores = []
        test_f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            params = {
                'boosting_type': 'gbdt',
                'num_leaves': 31,  # Default
                'n_estimators': 100,  # Default
                'max_depth': -1,  # Default
                'min_child_samples': 20,  # Default
                'subsample': 1.0,  # Default
                'colsample_bytree': 1.0,  # Default
                'scale_pos_weight': 1,  # Default
                'reg_alpha': 0,  # Default
                'reg_lambda': 0,  # Default
                'random_state': 123
            }
            params[param_name] = value
            
            model = LGBMClassifier(**params)
            
            start_time = time.time()
            try:
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                y_test_pred = model.predict(X_test)
                
                train_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
                test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
            except Exception as e:
                print(f"Error while training or predicting with {param_name}={value}: {e}")

        if train_f1_scores and test_f1_scores:
            results[param_name]['train_f1_scores'].append(np.mean(train_f1_scores))
            results[param_name]['test_f1_scores'].append(np.mean(test_f1_scores))
        else:
            print(f"No scores collected for {param_name} with value {value}")

# Evaluate each hyperparameter
for param_name, param_values in hyperparameters.items():
    evaluate_hyperparameter(param_name, param_values)

# Plotting results for each hyperparameter
for param_name in hyperparameters:
    if results[param_name]['train_f1_scores']:
        plt.figure(figsize=(12, 6))
        plt.plot(hyperparameters[param_name], results[param_name]['train_f1_scores'], marker='o', label='Train F1 Score')
        plt.plot(hyperparameters[param_name], results[param_name]['test_f1_scores'], marker='o', label='Test F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Macro F1 Score vs. {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No results for hyperparameter {param_name}")

# Print results
for param_name in hyperparameters:
    print(f"Hyperparameter: {param_name}")
    print(f"Values: {hyperparameters[param_name]}")
    print(f"Train F1 Scores: {results[param_name]['train_f1_scores']}")
    print(f"Test F1 Scores: {results[param_name]['test_f1_scores']}")
    print()

    import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values
y = data['label'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# List of hyperparameters and their ranges
hyperparameters = {
    'num_leaves': [31, 40, 50, 63, 70, 80, 90, 100],
    'n_estimators': [75, 100, 125, 150, 175, 200],
    'max_depth': [-1, 10, 20, 30, 40, 50, 60, 70],
    'min_child_samples': [1, 4, 8, 12, 16, 20],
    'scale_pos_weight': [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
    'reg_alpha': [0, 0.4, 0.8, 1.2, 1.6, 2],
    'reg_lambda': [0, 0.4, 0.8, 1.2, 1.6, 2]
}

# Initialize a dictionary to store results
results = {param: {'train_f1_scores': [], 'test_f1_scores': []} for param in hyperparameters}

def evaluate_hyperparameter(param_name, param_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    for value in param_values:
        print(f"Evaluating {param_name} with value {value}...")
        train_f1_scores = []
        test_f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            params = {
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # Default
                'n_estimators': 100,  # Default
                'max_depth': -1,  # Default
                'min_child_samples': 10,  # Default
                'subsample': 1.0,  # Default
                'colsample_bytree': 1.0,  # Default
                'scale_pos_weight': 1,  # Default
                'reg_alpha': 0,  # Default
                'reg_lambda': 0,  # Default
                'random_state': 123
            }
            params[param_name] = value
            
            model = LGBMClassifier(**params)
            
            start_time = time.time()
            try:
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                y_test_pred = model.predict(X_test)
                
                train_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
                test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
            except Exception as e:
                print(f"Error while training or predicting with {param_name}={value}: {e}")

        if train_f1_scores and test_f1_scores:
            results[param_name]['train_f1_scores'].append(np.mean(train_f1_scores))
            results[param_name]['test_f1_scores'].append(np.mean(test_f1_scores))
        else:
            print(f"No scores collected for {param_name} with value {value}")

# Evaluate each hyperparameter
for param_name, param_values in hyperparameters.items():
    evaluate_hyperparameter(param_name, param_values)

# Plotting results for each hyperparameter
for param_name in hyperparameters:
    if results[param_name]['train_f1_scores']:
        plt.figure(figsize=(12, 6))
        plt.plot(hyperparameters[param_name], results[param_name]['train_f1_scores'], marker='o', label='Train F1 Score')
        plt.plot(hyperparameters[param_name], results[param_name]['test_f1_scores'], marker='o', label='Test F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Macro F1 Score vs. {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No results for hyperparameter {param_name}")

# Print results
for param_name in hyperparameters:
    print(f"Hyperparameter: {param_name}")
    print(f"Values: {hyperparameters[param_name]}")
    print(f"Train F1 Scores: {results[param_name]['train_f1_scores']}")
    print(f"Test F1 Scores: {results[param_name]['test_f1_scores']}")
    print()

    import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values
y = data['label'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# List of hyperparameters and their ranges
hyperparameters = {
    'num_leaves': [31, 40, 50, 63, 70, 80, 90, 100, 110, 120, 130],
    'n_estimators': [75, 85, 95, 105, 115, 125],
    'max_depth': [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_child_samples': [1, 2, 4, 6, 8, 10],
    'reg_lambda': [1, 1.2, 1.4, 1.6, 1.8, 2.0]
}

# Initialize a dictionary to store results
results = {param: {'train_f1_scores': [], 'test_f1_scores': []} for param in hyperparameters}

def evaluate_hyperparameter(param_name, param_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    for value in param_values:
        print(f"Evaluating {param_name} with value {value}...")
        train_f1_scores = []
        test_f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            params = {
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # Default
                'n_estimators': 100,  # Default
                'max_depth': -1,  # Default
                'min_child_samples': 10,  # Default
                'subsample': 1.0,  # Default
                'colsample_bytree': 1.0,  # Default
                'scale_pos_weight': 1.5,  # DONE
                'reg_alpha': 1.2,  # DONE
                'reg_lambda': 0,  # Default
                'random_state': 123
            }
            params[param_name] = value
            
            model = LGBMClassifier(**params)
            
            start_time = time.time()
            try:
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                y_test_pred = model.predict(X_test)
                
                train_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
                test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
            except Exception as e:
                print(f"Error while training or predicting with {param_name}={value}: {e}")

        if train_f1_scores and test_f1_scores:
            results[param_name]['train_f1_scores'].append(np.mean(train_f1_scores))
            results[param_name]['test_f1_scores'].append(np.mean(test_f1_scores))
        else:
            print(f"No scores collected for {param_name} with value {value}")

# Evaluate each hyperparameter
for param_name, param_values in hyperparameters.items():
    evaluate_hyperparameter(param_name, param_values)

# Plotting results for each hyperparameter
for param_name in hyperparameters:
    if results[param_name]['train_f1_scores']:
        plt.figure(figsize=(12, 6))
        plt.plot(hyperparameters[param_name], results[param_name]['train_f1_scores'], marker='o', label='Train F1 Score')
        plt.plot(hyperparameters[param_name], results[param_name]['test_f1_scores'], marker='o', label='Test F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Macro F1 Score vs. {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No results for hyperparameter {param_name}")

# Print results
for param_name in hyperparameters:
    print(f"Hyperparameter: {param_name}")
    print(f"Values: {hyperparameters[param_name]}")
    print(f"Train F1 Scores: {results[param_name]['train_f1_scores']}")
    print(f"Test F1 Scores: {results[param_name]['test_f1_scores']}")
    print()

    import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values
y = data['label'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# List of hyperparameters and their ranges
hyperparameters = {
    'min_child_samples': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'colsample_bytree': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
}

# Initialize a dictionary to store results
results = {param: {'train_f1_scores': [], 'test_f1_scores': []} for param in hyperparameters}

def evaluate_hyperparameter(param_name, param_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    for value in param_values:
        print(f"Evaluating {param_name} with value {value}...")
        train_f1_scores = []
        test_f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            params = {
                'boosting_type': 'gbdt',
                'num_leaves': 70,
                'n_estimators': 85,
                'max_depth': 30,
                'min_child_samples': 2,
                'subsample': 1.0,  # Default
                'colsample_bytree': 1.0,  # Default
                'scale_pos_weight': 1.5,
                'reg_alpha': 1.2,
                'reg_lambda': 1,
                'random_state': 123
            }
            params[param_name] = value
            
            model = LGBMClassifier(**params)
            
            start_time = time.time()
            try:
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                y_test_pred = model.predict(X_test)
                
                train_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
                test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
                
                # Check the impact on training
                print(f"Trained with {param_name}={value} in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error while training or predicting with {param_name}={value}: {e}")

        if train_f1_scores and test_f1_scores:
            results[param_name]['train_f1_scores'].append(np.mean(train_f1_scores))
            results[param_name]['test_f1_scores'].append(np.mean(test_f1_scores))
        else:
            print(f"No scores collected for {param_name} with value {value}")

# Evaluate each hyperparameter
for param_name, param_values in hyperparameters.items():
    evaluate_hyperparameter(param_name, param_values)

# Plotting results for each hyperparameter
for param_name in hyperparameters:
    if results[param_name]['train_f1_scores']:
        plt.figure(figsize=(12, 6))
        plt.plot(hyperparameters[param_name], results[param_name]['train_f1_scores'], marker='o', label='Train F1 Score')
        plt.plot(hyperparameters[param_name], results[param_name]['test_f1_scores'], marker='o', label='Test F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Macro F1 Score vs. {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No results for hyperparameter {param_name}")

# Print results
for param_name in hyperparameters:
    print(f"Hyperparameter: {param_name}")
    print(f"Values: {hyperparameters[param_name]}")
    print(f"Train F1 Scores: {results[param_name]['train_f1_scores']}")
    print(f"Test F1 Scores: {results[param_name]['test_f1_scores']}")
    print()

    import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X = data.drop(['label', 'id'], axis=1).values
y = data['label'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# List of hyperparameters and their ranges
hyperparameters = {
    'colsample_bytree': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
}

# Initialize a dictionary to store results
results = {param: {'train_f1_scores': [], 'test_f1_scores': []} for param in hyperparameters}

def evaluate_hyperparameter(param_name, param_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    for value in param_values:
        print(f"Evaluating {param_name} with value {value}...")
        train_f1_scores = []
        test_f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            params = {
                'boosting_type': 'gbdt',
                'num_leaves': 70,
                'n_estimators': 85,
                'max_depth': 30,
                'min_child_samples': 4,
                'subsample': 1.0,  # Default
                'colsample_bytree': 1.0,  # Default
                'scale_pos_weight': 1.5,
                'reg_alpha': 1.2,
                'reg_lambda': 1,
                'random_state': 123
            }
            params[param_name] = value
            
            model = LGBMClassifier(**params)
            
            start_time = time.time()
            try:
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                y_test_pred = model.predict(X_test)
                
                train_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
                test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
                
                # Check the impact on training
                print(f"Trained with {param_name}={value} in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error while training or predicting with {param_name}={value}: {e}")

        if train_f1_scores and test_f1_scores:
            results[param_name]['train_f1_scores'].append(np.mean(train_f1_scores))
            results[param_name]['test_f1_scores'].append(np.mean(test_f1_scores))
        else:
            print(f"No scores collected for {param_name} with value {value}")

# Evaluate each hyperparameter
for param_name, param_values in hyperparameters.items():
    evaluate_hyperparameter(param_name, param_values)

# Plotting results for each hyperparameter
for param_name in hyperparameters:
    if results[param_name]['train_f1_scores']:
        plt.figure(figsize=(12, 6))
        plt.plot(hyperparameters[param_name], results[param_name]['train_f1_scores'], marker='o', label='Train F1 Score')
        plt.plot(hyperparameters[param_name], results[param_name]['test_f1_scores'], marker='o', label='Test F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Macro F1 Score vs. {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No results for hyperparameter {param_name}")

# Print results
for param_name in hyperparameters:
    print(f"Hyperparameter: {param_name}")
    print(f"Values: {hyperparameters[param_name]}")
    print(f"Train F1 Scores: {results[param_name]['train_f1_scores']}")
    print(f"Test F1 Scores: {results[param_name]['test_f1_scores']}")
    print()

    import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

# Load your training dataset
train_data = pd.read_csv('data/train_tfidf_features.csv')

# Exclude the first column and use the second column as the label
X_train = train_data.drop(['label', 'id'], axis=1).values
y_train = train_data['label'].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Load your test dataset
test_data = pd.read_csv('data/test_tfidf_features.csv')

# Exclude the first column
X_test = test_data.drop(['id'], axis=1).values

print(f"X_test shape: {X_test.shape}")

# Define model parameters
params = {
    'boosting_type': 'gbdt',
    'num_leaves': 70,
    'n_estimators': 85,
    'max_depth': 30,
    'min_child_samples': 4,
    'subsample': 1.0,  # Default
    'colsample_bytree': 0.55,
    'scale_pos_weight': 1.5,
    'reg_alpha': 1.2,
    'reg_lambda': 1,
    'random_state': 123
}

# Train the model on the entire training dataset
model = LGBMClassifier(**params)
start_time = time.time()
model.fit(X_train, y_train)
print(f"Model trained in {time.time() - start_time:.2f} seconds")

# Make predictions on the test dataset
y_test_pred = model.predict(X_test)

# Output the predictions
output = pd.DataFrame({'id': test_data['id'], 'label': y_test_pred})
#output.to_csv('Optimal_boosting.csv', index=False) #uncomment to get file, otherwise file is found in /task3_predictions


print("Predictions saved to Optimal_boosting.csv")