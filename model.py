# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:35:43 2024

@author: luked
"""

# imports
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import csv
import pandas as pd
import numpy as np #numerical
import seaborn as sns #plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set resolution of plotted figures
plt.rcParams['figure.dpi'] = 200

# Configure Seaborn plot styles: Set background color and use dark grid
sns.set(rc={'axes.facecolor': '#faded8'}, style='darkgrid')


# =============================================================================
# Question 1  DATABASING
# =============================================================================


# Create a database connection
conn_heart_db = sqlite3.connect('heart.db') # heart.db will be created on launch
print("connection success")

# declare cursor variable
cursor = conn_heart_db.cursor()    # cursor is used to execute SQL statements

# check if heart table exists
listOfTables = cursor.execute(
  """SELECT name FROM sqlite_master WHERE type='table'
  AND name='heart'; """).fetchall()
 
if listOfTables != []:
    # drop SQL table in case it already exists
    cursor.execute('''
        DROP TABLE heart''')
    print("heart table dropped and no longer exists")



# create SQL table 
cursor.execute( '''
    CREATE TABLE heart(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex INTEGER,
        cp INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        fbs ITEGER,
        restecg INTEGER,
        thalach INTEGER,
        exang INTEGER,
        oldpeak REAL,
        slope INTEGER,
        ca INTEGER,
        thal INTEGER,
        target INTEGER
        )''')

print("Table 'heart' created")

## INSERT DATA 
# SQL query to insert data into the heart table
insert_records = f"INSERT INTO heart (age, sex, cp, trestbps, chol, fbs, \
                                    restecg, thalach, exang, oldpeak, \
                                    slope, ca, thal, target) \
                                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
 
# opening the heart.csv file
with open('heart.csv', newline='') as file:
    data_contents = csv.reader(file, delimiter=';')
    
    # skip header row
    next(data_contents)
    
    # importing the contents of the file into our heart table
    cursor.executemany(insert_records, data_contents)

# verify that the data was inserted into the heart table
select_all = "SELECT * FROM heart"
rows = cursor.execute(select_all).fetchall()
# output to the console screen
for records in rows:
    print(records)

# commit the changes to the heart.db database
conn_heart_db.commit()

# Read data from heart table in heart.db database
df = pd.read_sql_query("SELECT * from heart", conn_heart_db)


# =============================================================================
# READING DATA FROM CSV
# =============================================================================


# Reading heart data from heart.csv
# link to csv file
csv_path = 'heart.csv'

# read csv file (specify seperator in the csv file)
df = pd.read_csv(csv_path, sep=';')   

# # Read data from heart table in heart.db database
# df = pd.read_sql_query("SELECT * from heart", conn_heart_db)


# =============================================================================
# Data Cleaning Preparation
# =============================================================================

## Rename column names
# rename the columns incase headings contain symbols
df.rename(columns={
    'age': 'age',
    'sex': 'sex',
    'cp': 'cp',
    'trestbps': 'trestbps',
    'chol': 'chol',
    'fbs': 'fbs',
    'restecg': 'restecg',
    'thalach': 'thalach',
    'exang': 'exang',
    'oldpeak': 'oldpeak',
    'slope': 'slope',
    'ca': 'ca',
    'thal': 'thal',
    'target': 'target'
}, inplace=True)    # use inplace to apply the changes to the dataframe


## Datatypes
# check datatypes
df.dtypes

# Need to convert non numeric columns (categorical columns) to object datatype
# Define the numeric/continuous features
numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Identify columns to be converted to object data type
columns_to_convert = [column for column in df.columns if column not in numeric_columns]

# Convert the identified columns to object data type
df[columns_to_convert] = df[columns_to_convert].astype('object')

df.dtypes


## Missing values
# Check if there are any missing values
df.isna().sum()


## Duplicates
# check if any of the data is duplicated
df.loc[df.duplicated()]

# Investigate reasons for duplicate row
df.query('age == 38 and trestbps == 138')

# Drop the duplicate row with id: 164 and reset_index() drop the old index column and therefore continue the index numbering
df = df.drop(index=164).reset_index(drop=True)

# Check if there are any duplicates (there should be none)
df.loc[df.duplicated()] # comment: no duplicates are returned


## Check for Data Characters Mistakes
# Check for 0 ‘unknown’ data in 'thal'
# Get unique values in 'thal' column
df['thal'].unique()

# Count the occurrences  in each category
df['thal'].value_counts()

# Drop rows where 'thal' is 0
df = df[df['thal'] != 0]

# Get unique values in 'thal' column
df['thal'].unique()



# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Bivariate Analysis
# Categorical Data
# =============================================================================

# Filter out categorical features for the bivariate analysis
categorical_columns = df.columns.difference(numeric_columns)
# Remove 'target' variable from categorical_columns
categorical_columns = [column for column in categorical_columns if column != 'target']

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

for i,col in enumerate(categorical_columns):
    
    # Create a cross tabulation showing the proportion of purchased and non-purchased loans for each category of the feature
    cross_tab = pd.crosstab(index=df[col], columns=df['target'])
    
    # Using the normalize=True argument gives us the index-wise proportion of the data
    cross_tab_proportion = pd.crosstab(index=df[col], columns=df['target'], normalize='index')

    # Define colormap
    color_map = ListedColormap(['#00a5c9', 'blue'])
    
    # Plot the stacked bar charts
    x, y = i//4, i%4
    cross_tab_proportion.plot(kind='bar', ax=ax[x,y], stacked=True, width=0.9, colormap=color_map,
                        legend=False, ylabel='Proportion', sharey=True)
    
    # Add proportions and counts of individual bars to the plot
    for index, value in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_location) in zip(cross_tab_proportion.loc[value],cross_tab.loc[value],cross_tab_proportion.loc[value].cumsum()):
            ax[x,y].text(x=index-0.3, y=(y_location-proportion)+(proportion/2)-0.03,
                         s = f'    {count}\n({np.round(proportion * 100, 1)}%)', 
                         color = "black", fontsize=10, fontweight="bold")
    
    # Add legend
    ax[x,y].legend(title='target', loc=(0.7,0.9), fontsize=9, ncol=2)
    # Set y limit
    ax[x,y].set_ylim([0,1.12])
    # Rotate xticks
    ax[x,y].set_xticklabels(ax[x,y].get_xticklabels(), rotation=0)
    
# Set the title for the entire figure            
plt.suptitle('Distribution of Categorical Variables vs Target Variable Stacked Barplots', fontsize=24)
plt.tight_layout()                     
plt.show()  # Show plot



# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Bivariate Analysis
# Numeric/Continuous Data
# =============================================================================

# Set color palette
sns.set_palette(['#00a5c9', 'blue'])

# Create subplots
fig, ax = plt.subplots(len(numeric_columns), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})

# Loop through each continuous feature to create barplots and kde plots
for i, var in enumerate(numeric_columns):
    # Barplot showing mean value of variable for each target category
    graph = sns.barplot(data=df, x="target", y=var, ax=ax[i,0])
    
    # KDE plot showing the distribution of the feature for each target category
    sns.kdeplot(data=df[df["target"]==0], x=var, fill=True, linewidth=3, ax=ax[i,1], label='0')
    sns.kdeplot(data=df[df["target"]==1], x=var, fill=True, linewidth=3, ax=ax[i,1], label='1')
    ax[i,1].set_yticks([])
    ax[i,1].legend(title='Heart Disease', loc='upper right')
    
    # add mean values to barplot
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')
        
# Set the title for the entire figure
plt.suptitle('Distribution of Numeric Variables vs Target Variable', fontsize=24)
plt.tight_layout()                     
plt.show()  # Show plot


# =============================================================================
# QUESTION 3
# DATA PREPROCESSING
# =============================================================================

# check for nulls
df.isnull().sum().sum()


### Handling Outliers

# Calculate quantiles 1 and 3
Quantile1 = df[numeric_columns].quantile(0.25)
Quantile3 = df[numeric_columns].quantile(0.75)
IQR = Quantile3 - Quantile1 # Calculate IQR
# Calculate outlier count per numeric column
outlier_count_specified = ((df[numeric_columns] < (Quantile1 - 1.5 * IQR)) | (df[numeric_columns] > (Quantile3 + 1.5 * IQR))).sum()

# Output outlier count
outlier_count_specified



### Categorical Features Encoding

# Implementing one-hot encoding on the specified categorical features
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)

# Convert remaining categorical variables that don't require one-hot encoding to integer data type
columns_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
for column in columns_to_convert:
    df_encoded[column] = df_encoded[column].astype(int)

df_encoded.dtypes # get types


# Displaying the resulting DataFrame after one-hot encoding
df_encoded.head()



### Transforming Skewed Features

# Define the features (X) and the output labels (y)
X = df_encoded.drop('target', axis=1)
y = df_encoded['target'] 

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Adding a small constant to 'oldpeak' to make all values positive
X_train['oldpeak'] = X_train['oldpeak'] + 0.001
X_test['oldpeak'] = X_test['oldpeak'] + 0.001

# Check distribution of numeric columns
fig, ax = plt.subplots(2, 5, figsize=(15,15))

# Original Distributions
for i, column in enumerate(numeric_columns):
    sns.histplot(X_train[column], kde=True, ax=ax[0,i], color='#00a5c9').set_title(f'Original {column}')
    

# Applying Box-Cox Transformation
# Dictionary to store lambda values for each column
lambdas = {}

for i, column in enumerate(numeric_columns):
    # apply box-cox for positive values only
    if X_train[column].min() > 0:
        X_train[column], lambdas[column] = boxcox(X_train[column])
        # Applying the same lambda to test data
        X_test[column] = boxcox(X_test[column], lmbda=lambdas[column]) 
        sns.histplot(X_train[column], kde=True, ax=ax[1,i], color='blue').set_title(f'Transformed {column}')
    else:
        sns.histplot(X_train[column], kde=True, ax=ax[1,i], color='orange').set_title(f'{column} (Not Transformed)')

fig.tight_layout()
plt.show()  # show plot



# =============================================================================
# =============================================================================
# # QUESTION 3.2
# =============================================================================
# =============================================================================

#  Function to consolidate a models metrics into a dataframe
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the performance of a trained model on test data using various metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extracting metrics
    metrics = {
        "precision_0": report["0"]["precision"],
        "precision_1": report["1"]["precision"],
        "recall_0": report["0"]["recall"],
        "recall_1": report["1"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_test, y_pred)
    }
    
    # Convert dictionary to dataframe
    df = pd.DataFrame(metrics, index=[model_name]).round(2)
    
    return df

# =============================================================================
# DECISION TREE MODEL BUILDING
# =============================================================================

### BUILDING

# Define the base Decision Tree model
decisiontree_base = DecisionTreeClassifier(random_state=0)


# Decision Tree Hyperparameter Tuning
## Use GridSearchCV and cross-validation (StratifiedKFold) to exhaustively considers all parameter combinations of hyperparameters.
## The combination that has the greatest recall for class 1 is then selected as the default scoring metric. 

# Hyperparameter grid for Decision Tree
param_grid_decisiontree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,3],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

# Create cross-validation object with StratifiedKFold in order to ensure class distribution is same across all folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Create GridSearchCV object
clf = GridSearchCV(decisiontree_base, param_grid_decisiontree, cv=cv, scoring='recall', n_jobs=-1)

# Fit GridSearchCV object to training data
clf.fit(X_train, y_train)

# Get the best hyperparameters
best_decisiontree_hyperparams = clf.best_params_

# Get the best model that has been fitted to the training data
best_decisiontree = clf.best_estimator_

print('Decision Tree Optimal Hyperparameters: \n', best_decisiontree_hyperparams)



### Decision Tree Model EVALUATION

# Evaluate optimized model on train data
print(classification_report(y_train, best_decisiontree.predict(X_train)))

# Evaluate optimized model on test data
print(classification_report(y_test, best_decisiontree.predict(X_test)))


# Consolidate Decision Tree metrics into dataframe by calling evaluate_model function
decisiontree_evaluation = evaluate_model(best_decisiontree, X_test, y_test, 'Decision Tree')
decisiontree_evaluation # output consolidated results



# =============================================================================
# RANDOM FOREST MODEL BUILDING
# =============================================================================

### BUILDING

# Define the Random Forest Model
randomforest_base = RandomForestClassifier(random_state=0)

# Random Forest Hyperparameter Tuning
## Use GridSearchCV and cross-validation (StratifiedKFold) to exhaustively considers all parameter combinations of hyperparameters.
## The combination that has the greatest recall for class 1 is then selected as the default scoring metric. 

# Hyperparameter grid for Random Forest
param_grid_randomforest = {
    'n_estimators': [10, 30, 50, 70, 90],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,3],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Create cross-validation object with StratifiedKFold in order to ensure class distribution is same across all folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Create GridSearchCV object
clf = GridSearchCV(randomforest_base, param_grid_randomforest, cv=cv, scoring='recall', n_jobs=-1)

# Fit GridSearchCV object to training data
clf.fit(X_train, y_train)

# Get the best hyperparameters
best_randomforest_hyperparams = clf.best_params_

# Get the best model that has been fitted to the training data
best_randomforest = clf.best_estimator_

print('Random Forest Optimal Hyperparameters: \n', best_randomforest_hyperparams)


### Random Forest Model EVALUATION


# Evaluate optimized model on train data
print(classification_report(y_train, best_randomforest.predict(X_train)))

# Evaluate optimized model on test data
print(classification_report(y_test, best_randomforest.predict(X_test)))

# Consolidate Random Forest metrics into dataframe by calling evaluate_model function
randomforest_evaluation = evaluate_model(best_randomforest, X_test, y_test, 'Random Forest')
randomforest_evaluation # output consolidated results


# =============================================================================
# SVM MODEL BUILDING
# =============================================================================

### BUILDING

# Define the SVM Model and setup pipeline with scaling
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True)) 
])


# SWM Hyperparameter Tuning
## Use GridSearchCV and cross-validation (StratifiedKFold) to exhaustively considers all parameter combinations of hyperparameters.
## The combination that has the greatest recall for class 1 is then selected as the default scoring metric. 

# Hyperparameter grid for SVM
param_grid_svm = {
    'svm__C': [0.0011, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.5, 1, 5],  
    'svm__degree': [2, 3, 4]
}
 
# Create cross-validation object with StratifiedKFold in order to ensure class distribution is same across all folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Create GridSearchCV object
clf = GridSearchCV(svm_pipeline, param_grid_svm, cv=cv, scoring='recall', n_jobs=-1)

# Fit GridSearchCV object to training data
clf.fit(X_train, y_train)

# Get the best hyperparameters
best_svm_hyperparams = clf.best_params_

# Get the best model that has been fitted to the training data
best_svm = clf.best_estimator_

print('Random Forest Optimal Hyperparameters: \n', best_svm_hyperparams)


### SVM Model EVALUATION


# Evaluate optimized model on train data
print(classification_report(y_train, best_svm.predict(X_train)))

# Evaluate optimized model on test data
print(classification_report(y_test, best_svm.predict(X_test)))

# Consolidate SVM metrics into dataframe by calling evaluate_model function
svm_evaluation = evaluate_model(best_svm, X_test, y_test, 'SVM')
svm_evaluation # output consolidated results



# =============================================================================
# BEST MODEL EVALUATION
# =============================================================================

# Concatenate all the evaluation dataframes
all_evaluations = [decisiontree_evaluation, randomforest_evaluation, svm_evaluation]
evaluation_results = pd.concat(all_evaluations)

# Sort results by 'recall_1' (true positives)
evaluation_results = evaluation_results.sort_values(by='recall_1', ascending=False).round(2)
evaluation_results # display all evaluation results


## EVALUATION BASED ON RECALL_1

# Sort values based on 'recall_1'
evaluation_results.sort_values(by='recall_1', ascending=True, inplace=True)
recall_1_scores = evaluation_results['recall_1']

# Plot the horizontal bar chart
fig, ax = plt.subplots(figsize=(15, 10), dpi=80)
ax.barh(evaluation_results.index, recall_1_scores, color='darkturquoise')

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(recall_1_scores, evaluation_results.index)):
    ax.text(value + 0.01, i, f"{value:.2f}", ha='left', va='center', fontweight='bold', color='darkturquoise', fontsize=20)
    ax.text(0.1, i, name, ha='left', va='center', fontweight='bold', color='white', fontsize=25)

# Remove the yticks
ax.set_yticks([])

# Set limit for the x-axis 
ax.set_xlim([0, 1.1])

# Add a title and an xlabel
plt.title("Recall for Positive Class across Models", fontweight='bold', fontsize=25)
plt.xlabel('Recall Value', fontsize=15)
plt.show()



## EVALUATION BASED ON ACCURACY

# Sort results by 'accuracy'
evaluation_results = evaluation_results.sort_values(by='accuracy', ascending=False).round(2)
evaluation_results # display all evaluation results


# Sort values based on 'accuracy'
evaluation_results.sort_values(by='accuracy', ascending=True, inplace=True)
accuracy_scores = evaluation_results['accuracy']

# Plot the horizontal bar chart
fig, ax = plt.subplots(figsize=(15, 10), dpi=80)
ax.barh(evaluation_results.index, accuracy_scores, color='darkturquoise')

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(accuracy_scores, evaluation_results.index)):
    ax.text(value + 0.01, i, f"{value:.2f}", ha='left', va='center', fontweight='bold', color='darkturquoise', fontsize=20)
    ax.text(0.1, i, name, ha='left', va='center', fontweight='bold', color='white', fontsize=25)

# Remove the yticks
ax.set_yticks([])

# Set limit for the x-axis 
ax.set_xlim([0, 1.1])

# Add a title and an xlabel
plt.title("Accuracy Score across Models", fontweight='bold', fontsize=25)
plt.xlabel('Accuracy Value', fontsize=15)
plt.show()



# =============================================================================
# DOWNLOAD MODEL
# =============================================================================

# save model to disk
joblib.dump(best_svm, "svm_heart_disease_model.sav")
# Assuming 'lambdas' is a dictionary containing the lambdas for each column
joblib.dump(lambdas, "boxcox_lambdas.pkl")
