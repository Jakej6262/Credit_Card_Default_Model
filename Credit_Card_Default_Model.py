#import necessary libraries
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Access Data from csv file
Data=pd.read_csv(r'Credit Card Default Modeling Data-Use This.csv' ,low_memory=True)

try:
    Unprocessed_Df=pd.DataFrame(data=Data)
    logger.info("Data loaded successfully")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

#check for null values in the dataset
logger.info("Checking for null values in the dataset")
print(Unprocessed_Df.isnull().sum())

logger.info("Processing the data")
#convert necessary columns to proper data types
numeric_columns=["X1","X5"]
for i in range(12,24):
    numeric_columns.append("X"+str(i))
Unprocessed_Df[numeric_columns]=Unprocessed_Df[numeric_columns].apply(pd.to_numeric, errors='coerce')

X_columns=[column for column in Unprocessed_Df.columns.values]

Processed_df=pd.get_dummies(Unprocessed_Df,columns=["X3","X4"],drop_first=True) 

#view column types
print(Processed_df.dtypes)

# shuffle the data set for splitting
Processed_df = Processed_df.sample(frac=1, random_state=42)
print(Processed_df.head())
# pf.drop(columns=["ID"], inplace=True)


#view heatmap of the data set
plt.figure(figsize=(20,20))
sns.heatmap(Processed_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap")
plt.show()


logger.info("Resampling to deal with class imbalance")
#Show correlations between features and target variable
plt.figure(figsize=(20,20))
sns.heatmap(Processed_df.corr()[["Y"]].sort_values(by="Y", ascending=False), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
# Rotate X and Y axis labels
plt.xticks(rotation=45)   # Rotates column names
plt.yticks(rotation=0) 
plt.title("Correlation Heatmap with Target Variable")
plt.show()

# Credit Limit Distribution
plt.figure(figsize=(10, 6))
sns.histplot(Processed_df['X1'], bins=30, kde=True)
plt.title('Distribution of Credit Limit (X1)')
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain')  # Avoid scientific notation
plt.grid(True)
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(Processed_df['X5'], bins=30, kde=True, color='orange')
plt.title('Distribution of Age (X5)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
#balance data set to deal with class imbalance
smote = SMOTE(random_state=42)
Features=Processed_df.drop(columns=["Y"])
Target=Processed_df["Y"]
X_resampled, y_resampled = smote.fit_resample(Features, Target)

#split columns into training and testing
logger.info("Splitting the data into training, testing, and validation sets")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)  

#split the test data into 2 sets for validation and testing
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Train Model using Decision Tree Classifier
logger.info("Training the Decision Tree Classifier")
dec_tree_model=DecisionTreeClassifier()
dec_tree_model.fit(X_train, y_train)
validation_predictions = dec_tree_model.predict(X_val)
logger.info("Evaluating the model on validation data")

#View initial results
print("Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_val, validation_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_val, validation_predictions))
print("Classification Report:\n", classification_report(y_val, validation_predictions))

#Tune the Decision Tree Classifier for recall
print("Tuning the Decision Tree Classifier for recall")
dec_tree_model = DecisionTreeClassifier(random_state=42)
param_grid = { 'max_depth': [3, 5, 7, 9], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4] }
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(dec_tree_model, param_grid, scoring='recall', cv=5)
grid_search.fit(X_train, y_train)
best_decision_tree_model = grid_search.best_estimator_
validation_predictions = best_decision_tree_model.predict(X_val)
logger.info("Evaluating the tuned model on validation data")
#View initial results
print("Tuned Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_val, validation_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_val, validation_predictions))
print("Classification Report:\n", classification_report(y_val, validation_predictions))
#evaluate the tuned model on the test set
test_predictions = best_decision_tree_model.predict(X_test)
print("Tuned Decision Tree Classifier Test Set Metrics:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))

# Train Model using Random Forest Classifier
logger.info("Training the Random Forest Classifier")
model=RandomForestClassifier()
model.fit(X_train, y_train)
validation_predictions = model.predict(X_val)
logger.info("Evaluating the model on validation data")

#View initial results
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_val, validation_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_val, validation_predictions))
print("Classification Report:\n", classification_report(y_val, validation_predictions))

y_probs = model.predict_proba(X_val)[:,1]  
precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
#View thresholds
Threshold_df=pd.DataFrame ({
    "Thresholds":thresholds,
    "Precision": precision[:-1],  
    'Recall': recall[:-1]
})

#filter for precision values greater than 0.8
Threshold_df=Threshold_df[Threshold_df["Precision"]>0.8]

PrecisionRecallDisplay.from_predictions(y_val, y_probs)
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

#plot ROC curve
plt.figure(figsize=(10, 6))
fpr, tpr, thresholds = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Predict probabilities
y_probs_val = model.predict_proba(X_val)[:, 1]
y_probs_test = model.predict_proba(X_test)[:, 1]

# Apply threshold
optimal_threshold = 0.41
val_predictions = (y_probs_val >= optimal_threshold).astype(int)
test_predictions = (y_probs_test >= optimal_threshold).astype(int)

# Evaluate
print("Random Forest Validation Set Metrics:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))

print("Random Forest Test Set Metrics:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))

# # Get feature importances
# importances = model.feature_importances_

# # Match them to feature names
# feature_importance_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Display top features
# print(feature_importance_df.head(23))

# #visualize features
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(23))
# plt.title('Top Feature Importances from Random Forest')
# plt.tight_layout()
# plt.show()


