#!/usr/bin/env python
# coding: utf-8

# In[1]:
pip install xgboost
pip install plotly
pip install matplotlib
pip install statsmodels
pip install tabulate


# In[6]:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import os

# In[7]:
# Check folder path for calling input data and store output data
os.getcwd()


# In[8]:
# Check and establish folder path for input and output data
os.chdir()


# In[10]:
# Input data path
data = pd.read_csv()
data

# In[11]:
# Create a DataFrame
dataframe = pd.DataFrame(data)
dataframe

# In[12]:
# Assuming your DataFrame is named 'df'
df_columns = dataframe.columns

# Display column names and their corresponding numbers
for i, column_name in enumerate(df_columns):
    print(f"Column {i+1}: {column_name}")

# In[13]:
# Subset dataset for a certain time point
sel_col_df = dataframe.iloc[:, [15] + list(range(45, 63)) + list(range(64, 66))]
# Display the DataFrame with the selected columns
print(sel_col_df)

# In[13]:
# Create a new DataFrame with the selected columns
df = pd.DataFrame(sel_col_df)
# Display the new DataFrame
print(df)

# In[14]:
# DATASET PARTITIONING
X = df.drop("physmature", axis=1)
y = df["physmature"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[15]:
# RANDOM FOREST REGRESSION MODE HYPERPARAMETER TUNING

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are deemed independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Split the data into training and testing sets into 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Random Forest Regressor
random_forest_model = RandomForestRegressor()

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=kf, n_jobs=-1, scoring='neg_mean_squared_error')
# Fit the model to find the best hyperparameters
grid_search.fit(X_train_scaled, y_train)
# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(y_test, y_pred)
print(f'Pearson Correlation Coefficient: {pearson_corr}')
# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')
# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(f'R-squared with best hyperparameters: {r_squared}')


# In[15]:
# RANDOM FOREST REGRESSOR AT CERTAIN TIME POINT USING BEST HYPERPARAMETERS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Standardize the features if needed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the Random Forest Regressor with the best hyperparameter.
# Example hyperparameter values shown below.
best_hyperparameters_RF = {'max_depth': 10,
                           'min_samples_leaf': 2,
                           'min_samples_split': 5,
                           'n_estimators': 100}

rf_best_model = RandomForestRegressor(**best_hyperparameters_RF)

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize an array to store feature importances
rf_feature_importances = np.zeros(X.shape[1])

# Perform cross-validated predictions
for train_index, _ in kf.split(X_scaled, y):
    X_train, y_train = X_scaled[train_index], y.iloc[train_index]
    rf_best_model.fit(X_train, y_train)
    rf_feature_importances += rf_best_model.feature_importances_

# Average feature importances across folds
rf_feature_importances /= kf.get_n_splits()

# Sort feature importances in ascending order
sorted_indices = np.argsort(rf_feature_importances)

# Plot the feature importances with 'cividis' colormap
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(X.shape[1]),
              rf_feature_importances[sorted_indices],
              align="center",
              color=plt.cm.get_cmap('cividis')(np.linspace(0, 1, len(sorted_indices))),
              edgecolor='black')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[sorted_indices],
                   rotation=45, ha='right')
ax.set_xlabel("Feature")
ax.set_ylabel("Feature Importance")
ax.set_title("Random Forest Feature Importance")

plt.tight_layout()

# Save the figure to a file (e.g., PNG format)
plt.savefig('RandomForest_feature_ranking_TP1.tiff', dpi=300)

plt.show()

# In[36]:
# XGBOOST REGRESSION HYPERPARAMETER TUNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Split the dataset into training and testing sets with 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost Regressor
xgb_model = XGBRegressor()

# Define hyperparameter grid to search
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Get the best hyperparameters
best_params = grid_search.best_params_
# Train the model with the best hyperparameters on the full training set
best_xgb_model = XGBRegressor(**best_params)
best_xgb_model.fit(X_train, y_train)
# Predict on the test set
y_pred = best_xgb_model.predict(X_test)
# Evaluate the performance on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# Evaluate Model
r2 = r2_score(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)
# Print the best hyperparameters and evaluation metrics
print("Best Hyperparameters:", best_params)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error on Test Set: {rmse}")
print(f'Pearson Correlation Coefficient: {pearson_corr}')
# Print R-squared value
print(f'R-squared: {r2}')


# In[37]:


# XGBOOST REGRESSION USING BEST HYPERPARAMETERS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Split the dataset into training and testing sets into 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost Regressor with the best hyperparameters.
# Example hyperparameter values shown below.
best_hyperparameters_XGBOOST = {'learning_rate': 0.1, 
                        'n_estimators': 50, 
                        'max_depth': 3, 
                        'min_child_weight': 1,
                        'subsample': 0.8,
                        'colsample_bytree': 1.0}

best_xgb_model = XGBRegressor(**best_hyperparameters_XGBOOST)

# Train the model on the full training set
best_xgb_model.fit(X_train, y_train)

# Plot feature importance in ascending order with 'cividis' colormap
xgb_feature_importance = best_xgb_model.feature_importances_ * 100  # Convert to percentage
sorted_indices = np.argsort(xgb_feature_importance)

# Plot the feature importances
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(
    range(X.shape[1]),
    xgb_feature_importance[sorted_indices],
    align="center",
    color=plt.cm.get_cmap('cividis')(np.linspace(0, 1, len(sorted_indices))),
    edgecolor='black'
)
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[sorted_indices], rotation=45, ha='right')
ax.set_xlabel("Feature")
ax.set_ylabel("Feature Importance (%)")
ax.set_title("XGBoost Feature Importance")

# Optionally, you can add numerical values on top of the bars
for i, v in enumerate(xgb_feature_importance[sorted_indices]):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save the figure to a file (e.g., PNG format)
plt.savefig('XGBoost_feature_ranking_TP0815_02032024.tiff')

plt.show()


# ADA BOOST REGRESSION HYPERPARAMETER TUNING

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Standardize the features if needed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets into 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the AdaBoost Regressor
adaboost = AdaBoostRegressor()

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
}

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(adaboost, param_grid, scoring='neg_mean_squared_error', cv=kf)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_adaboost_model = AdaBoostRegressor(**best_params)
best_adaboost_model.fit(X_train, y_train)

# Make predictions
y_pred = best_adaboost_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)
print("Root Mean Square Error:",rmse)
print("R^2 Score:", r2)


# ADA BOOST REGRESSION FITTING USING BEST HYPERPARAMETERS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Standardize the features if needed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the AdaBoost Regressor with the best hyperparameters
best_hyperparameters = {'n_estimators': 200, 'learning_rate': 0.1}
adaboost_best_model = AdaBoostRegressor(**best_hyperparameters)

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize an array to store feature importances
adaboost_feature_importances = np.zeros(X.shape[1])

# Perform cross-validated predictions
for train_index, _ in kf.split(X_scaled, y):
    X_train, _ = X_scaled[train_index], y[train_index]
    adaboost_best_model.fit(X_train, y[train_index])
    adaboost_feature_importances += adaboost_best_model.feature_importances_

# Average feature importances across folds
adaboost_feature_importances /= kf.get_n_splits()

# Sort feature importances in ascending order
sorted_indices = np.argsort(adaboost_feature_importances)

# Multiply y-axis values by 100
adaboost_feature_importances *= 1

# Plot the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(X.shape[1]), 
              adaboost_feature_importances[sorted_indices], 
              align="center", 
              color=plt.cm.get_cmap('cividis')(np.linspace(0, 1, len(sorted_indices))),
              edgecolor='black')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[sorted_indices], rotation=45, ha='right')
ax.set_xlabel("Feature")
ax.set_ylabel("Feature Importance (%)")
ax.set_title("AdaBoost Feature Importance")

# Save the figure to a file (e.g., PNG format)
plt.savefig('ADABoost_feature_ranking_TP1.tiff')
plt.show()


# In[46]:
# EXTRA TREES REGRESSION HYPERPARAMETER TUNING
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import scipy.stats

# Split the dataset into training and testing sets into 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Extra Trees Regressor
et_regressor = ExtraTreesRegressor()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(et_regressor, param_grid, scoring='r2', cv=5)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create an Extra Trees Regressor with the best hyperparameters
best_et_regressor = ExtraTreesRegressor(**best_params)

# Perform cross-validation to calculate evaluation metrics
cross_val_results = cross_val_score(best_et_regressor, X_train, y_train, cv=5, scoring='r2')
r2_mean = np.mean(cross_val_results)

# Train the model on the entire training set
best_et_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_et_regressor.predict(X_test)

# Calculate evaluation metrics on the test set
r_squared = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)

# Calculate Pearson correlation coefficient (r value)
pearson_corr, _ = scipy.stats.pearsonr(y_test, y_pred)

# Print the evaluation metrics
print("\nCross-Validation R-Square Mean:", r2_mean)
print("Test Set R-Square:", r_squared)
print("Test Set RMSE:", rmse)
print("Test Set MSE:", mse)
print("Test Set Pearson Correlation (r value):", pearson_corr)


# In[52]:
# EXTRA TREES REGRESSION FITTING USING BEST HYPERPARAMETERS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load your dataset (assuming it's in a DataFrame named df)
# Ensure that your target variable (dependent variable) is present in the dataset.
# Assuming "physmature" is your target variable, and other columns are independent variables.
# Adjust this according to your actual dataset structure.
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Split the data into training and testing sets using 80:20 split ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the best hyperparameters
best_hyperparameters_ExtraTrees = {'max_depth': 10,  
                        'min_samples_leaf': 4,
                        'min_samples_split': 10,
                        'n_estimators': 100}

# Initialize the Extra Trees Regressor with the best hyperparameters
extratrees_best_model = ExtraTreesRegressor(**best_hyperparameters_ExtraTrees)

# Train the model on the full dataset
extratrees_best_model.fit(X_scaled, y)

# Get feature importances
extratree_feature_importances = extratrees_best_model.feature_importances_

# Sort feature importances in ascending order
sorted_indices = np.argsort(extratree_feature_importances)

# Plot the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(X.shape[1]), extratree_feature_importances[sorted_indices], 
              align="center", 
              color=plt.cm.get_cmap('cividis')(np.linspace(0, 1, len(sorted_indices))),
              edgecolor='black')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[sorted_indices], rotation=90)
ax.set_xlabel("Feature")
ax.set_ylabel("Feature Importance")
ax.set_title("Extra Trees Feature Importance")

# Optionally, you can add numerical values on top of the bars
#for i, v in enumerate(extratree_feature_importances[sorted_indices]):
#    ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

# Calculate R-squared on the test set
y_pred = extratrees_best_model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print(f"R-squared on test set: {r_squared}")

# Save the figure to a file (e.g., PNG format)
plt.savefig('FFAR_FEATURE_IMPORTANCE_ExtraTrees_TP0815_02032024.tiff')

plt.show()


# In[16]:
# SUMMARY OF FEATURE IMPORTANCE RANKING OF ML MODELS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Assuming df is your DataFrame with features and target variable
X = df.drop("physmature", axis=1)
y = df["physmature"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features before splitting
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the best hyperparameters for each model
# Example hyperparameter values for each model is shown below.
best_hyperparameters_RF = {'max_depth': 10, 
                           'min_samples_leaf': 2, 
                           'min_samples_split': 5, 
                           'n_estimators': 100}
best_hyperparameters_XGBOOST = {'learning_rate': 0.1, 
                                'n_estimators': 50, 
                                'max_depth': 3, 
                                'min_child_weight': 1,
                                'subsample': 0.8, 
                                'colsample_bytree': 1.0}
best_hyperparameters_AdaBoost = {'n_estimators': 200, 
                                 'learning_rate': 1.0}
best_hyperparameters_ExtraTrees = {'max_depth': 30, 
                                   'min_samples_leaf': 4, 
                                   'min_samples_split': 10, 
                                   'n_estimators': 100}

# Initialize models with best hyperparameters
rf_model = RandomForestRegressor(**best_hyperparameters_RF)
xgb_model = XGBRegressor(**best_hyperparameters_XGBOOST)
adaboost_model = AdaBoostRegressor(**best_hyperparameters_AdaBoost)
extratrees_model = ExtraTreesRegressor(**best_hyperparameters_ExtraTrees)

# Fit models on the full training set
rf_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train, y_train)
adaboost_model.fit(X_train_scaled, y_train)
extratrees_model.fit(X_train_scaled, y_train)

# Get feature importances
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_
adaboost_importances = adaboost_model.feature_importances_
extratrees_importances = extratrees_model.feature_importances_

# Create a DataFrame with feature importances
feature_importance_df_0815 = pd.DataFrame({
    'Feature': X.columns,
    'RandomForest': rf_importances,
    'XGBoost': xgb_importances,
    'AdaBoost': adaboost_importances,
    'ExtraTrees': extratrees_importances
})

# Plot heatmap with adjusted cell dimensions
plt.figure(figsize=(12, 8))
heatmap_data = feature_importance_df_0815.set_index('Feature').transpose()
sns.heatmap(heatmap_data, cmap='OrRd', annot=True, fmt=".3f", linewidths=.5, annot_kws={"size": 8}, square=True, cbar_kws={"shrink": 0.8})
plt.title('Feature Importance Heatmap for Different Models')

# Save the figure to a file (e.g., PNG format)
plt.savefig('TP0815_feature_ranking_02042024.tiff', dpi=300)

plt.show()































