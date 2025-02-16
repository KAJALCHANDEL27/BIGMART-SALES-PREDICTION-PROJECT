# BIGMART-SALES-PREDICTION-PROJECT
### README

# BigMart Sales Prediction Project

This project involves predicting the sales of items at various BigMart outlets using machine learning techniques. The primary goal is to accurately predict sales by leveraging data preprocessing, exploratory data analysis (EDA), feature engineering, and model tuning.

## Project Structure

- `Train.csv`: Training dataset containing sales data.
- `Test.csv`: Test dataset for which predictions are to be made.
- `script.py`: Main script that includes data preprocessing, model training, tuning, and prediction.
- `subGB.csv`: Output file containing predictions for the test dataset.

## Installation

To run this project, you need the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Usage

1. **Load the Data**: Read the train and test datasets using pandas.
2. **Exploratory Data Analysis (EDA)**: Understand the data through descriptive statistics and visualizations.
3. **Data Preprocessing**:
   - Fill missing values.
   - Encode categorical variables.
   - Add new features (e.g., Outlet Age).
   - Standardize the data.
4. **Model Training and Evaluation**:
   - Split the data into training and validation sets.
   - Train an initial Gradient Boosting model.
   - Perform cross-validation to evaluate model performance.
   - Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
5. **Final Model Training and Prediction**:
   - Train the final model with the best hyperparameters on the entire training dataset.
   - Make predictions on the test dataset.
   - Ensure no negative values in the predictions.
6. **Save Predictions**: Save the predictions to a CSV file.

## Cross-Validation

Cross-validation is performed using 5-fold cross-validation to ensure the model's robustness. The cross-validation RMSE and its standard deviation are printed for better evaluation.

## Hyperparameter Tuning

Hyperparameter tuning is done using `GridSearchCV` or `RandomizedSearchCV` to find the best parameters for the Gradient Boosting model. The best parameters and cross-validation score are printed.

## Predictions

The final predictions are saved in the `subGB.csv` file, ensuring no negative values are present.

## Conclusion

This project demonstrates the end-to-end process of a machine learning project, from data preprocessing and EDA to model training, tuning, and prediction. It ensures robust model evaluation through cross-validation and hyperparameter tuning.

Feel free to run the provided script and make necessary adjustments based on your specific requirements.

### Detailed Report

## BigMart Sales Prediction Project: Detailed Report

### 1. Introduction

The objective of this project is to predict the sales of products at various BigMart outlets. Accurate sales predictions are crucial for inventory management, demand forecasting, and maximizing profits. This report outlines the complete workflow, from data preprocessing to model evaluation and tuning.

### 2. Data Loading and Exploration

The data is loaded from CSV files using pandas. Initial exploration is performed to understand the data structure, missing values, and statistical summaries.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

print(train.info())
print(train.describe())
print(train.shape)
```

### 3. Exploratory Data Analysis (EDA)

EDA involves visualizing the distribution of sales and the correlations between features. Histograms and heatmaps are used for this purpose.

```python
sns.histplot(train['Item_Outlet_Sales'], bins=50, kde=True)
plt.show()
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### 4. Data Preprocessing

Missing values are handled by filling them with appropriate statistics (mean, mode). Categorical variables are encoded using Label Encoding. A new feature, `Outlet_Age`, is added to the dataset.

```python
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type']:
    train[column] = le.fit_transform(train[column])
    test[column] = le.transform(test[column])

train['Outlet_Age'] = 2025 - train['Outlet_Establishment_Year']
test['Outlet_Age'] = 2025 - test['Outlet_Establishment_Year']
```

### 5. Feature Scaling

Standardization is applied to the features to bring them to a similar scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = train.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
X_test = test.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
```

### 6. Model Training and Evaluation

The data is split into training and validation sets. An initial Gradient Boosting model is trained and evaluated using cross-validation and validation data.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Train-Test Split
y = train['Item_Outlet_Sales']
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initial Gradient Boosting Model Training
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Cross-Validation
cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores)
print(f'Cross-Validation RMSE (Gradient Boosting): {rmse_cv.mean()} ± {rmse_cv.std()}')

# Validation
y_pred_gb = gb_model.predict(X_val)
rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
print(f'Validation RMSE (Gradient Boosting): {rmse_gb}')
```

### 7. Hyperparameter Tuning

Hyperparameters are tuned using `GridSearchCV` or `RandomizedSearchCV` to find the best parameters for the model.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_gradient_boosting(X_train, y_train, use_random_search=False):
    param_grid_gb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 1.0]
    }

    if use_random_search:
        search = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                                    param_distributions=param_grid_gb, 
                                    n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
    else:
        search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                              param_grid=param_grid_gb, 
                              cv=3, n_jobs=-1, verbose=2)

    start_time = time()
    search.fit(X_train, y_train)
    end_time = time()

    best_params_gb = search.best_params_
    best_model_gb = search.best_estimator_
    print(f'Best Parameters (Gradient Boosting): {best_params_gb}')
    print(f'Best Cross-Validation Score: {search.best_score_}')
    print(f'Time taken for search: {end_time - start_time:.2f} seconds')
    return best_model_gb

# Example usage
best_model_gb = tune_gradient_boosting(X_train, y_train, use_random_search=True)
```

### 8. Final Model Training and Prediction

The final model is trained with the best hyperparameters and predictions are made on the test dataset. Negative values in the predictions are handled by setting them to zero.

```python
# Final Model Training and Prediction using Gradient Boosting
best_model_gb.fit(X_scaled, y)
predictions = best_model_gb.predict(X_test_scaled)

# Ensure no negative values in predictions
predictions = np.maximum(predictions, 0)

test['Item_Outlet_Sales'] = predictions

# Save the predictions to a CSV file
GB = test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
GB.to_csv('subGB.csv', index=False)
print("Predictions saved to subGB.csv")
```

### 9. Conclusion

This project provides an end-to-end solution for sales prediction, including data preprocessing, model training, evaluation, and tuning. Cross-validation and hyperparameter tuning ensure the model's
