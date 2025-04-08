import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Random seed for reproducibility
RANDOM_SEED = 42

# Data ingestion from the CSV file into a pandas DataFrame
data = pd.read_csv("data/customer_churn.csv")

# Clean the column names in case there are extra spaces
data.columns = data.columns.str.strip()

# Printing the initial data to verify it is fetched correctly
print("Initial data preview")
print(data.head())

# Debug: Check the available column names
print("Available columns:", data.columns.tolist())

# Select categorical and numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns
numeric_cols = data.select_dtypes(include=['int64','float64']).columns

# Replace missing numeric values with the mean and categorical with 'missing'
imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='constant', fill_value='missing')

data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Encode the categorical columns using LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Optionally check correlation and distributions 
print('\nData summary:')
print(data.describe())

# Define features and target variable using the correct column name
# Assuming the column is named "Churn" (adjust if needed)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

# Build the machine learning pipeline 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=RANDOM_SEED))
])

# Define the hyperparameters grid for grid search
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [2, 5],
    'classifier__bootstrap': [True, False]
}

# Use StratifiedKFold to preserve class distributions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search to the training set
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best validation score:")
print(grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score: {:.4f}".format(roc_auc_score(y_test, y_pred_proba)))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Get important features driving predictions
importances = best_model.named_steps["classifier"].feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})
feature_importances = feature_importances.sort_values(by="importance", ascending=False)

print("\nFeature Importances:")
print(feature_importances.head(10))

# Visualize the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importances)
plt.title("Top Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
