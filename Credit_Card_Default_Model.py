
#Access Data from Excel sheet
import pandas as pd
Data=pd.read_excel(r"C:\Users\jakej\OneDrive\Desktop\Credit_Card_Default_Model\Credit Card Default Modeling Data-Use This.xlsx",sheet_name="Data")
Df=pd.DataFrame(data=Data)

#convert necessary columns to proper data types
numeric_columns=["X1","X5"]
for i in range(12,24):
    numeric_columns.append("X"+str(i))
Df[numeric_columns]=Df[numeric_columns].apply(pd.to_numeric, errors='coerce')

X_columns=[column for column in Df.columns.values]

pf=pd.get_dummies(Df,columns=["X3","X4"],drop_first=True) 
print(pf.columns.values)

#drop unique id column
pf.drop(columns=["ID"], inplace=True)

#make correlation matrix for the dataset
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = pf.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
#plt.show()

#split columns into training and testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
Features=pf.drop(columns=["Y"])
Target=pf["Y"]
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=42)  
#Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Load data into model
model=LogisticRegression(
class_weight='balanced')
model.fit(X_train_scaled, y_train)
#predict the target variable
predictions=model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))




