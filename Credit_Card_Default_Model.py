
#Access Data from Excel sheet
import pandas as pd
Data=pd.read_csv(r'Credit Card Default Modeling Data-Use This.csv' ,low_memory=True)
Df=pd.DataFrame(data=Data)

#check for null values in the dataset
print(Df.isnull().sum())

#convert necessary columns to proper data types
numeric_columns=["X1","X5"]
for i in range(12,24):
    numeric_columns.append("X"+str(i))
Df[numeric_columns]=Df[numeric_columns].apply(pd.to_numeric, errors='coerce')

X_columns=[column for column in Df.columns.values]

pf=pd.get_dummies(Df,columns=["X3","X4"],drop_first=True) 

pf.drop(columns=["ID"], inplace=True)

#view heatmap of the data set
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,20))
sns.heatmap(pf.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap")
plt.show()


#view class balance of y variable 
print(sum(pf["Y"].value_counts()))

#balance data set to deal with class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
Features=pf.drop(columns=["Y"])
Target=pf["Y"]
X_resampled, y_resampled = smote.fit_resample(Features, Target)


#split columns into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)  

#Standardize the data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Train Model using KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#params = { 'n_neighbors':[5,6,7,8,9,10], 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
#model=GridSearchCV(KNeighborsClassifier(), params, cv=5, scoring='recall')
model=RandomForestClassifier()
model.fit(X_train, y_train)
predictions=model.predict(X_test)

# Get feature importances
importances = model.feature_importances_

# Match them to feature names
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print(feature_importance_df.head(23))

#visualize features
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(23))
plt.title('Top 20 Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

#View initial results
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))



