
#Access Data from Excel sheet
import pandas as pd
Data=pd.read_excel(r"C:\Users\jakej\Downloads\Credit Card Default Modeling Data-Use This.xlsx",sheet_name="Data")
Df=pd.DataFrame(data=Data)
print(Df.info())

#convert necessary columns to proper data types
numeric_columns=["X1","X5"]
for i in range(12,24):
    numeric_columns.append("X"+str(i))
Df[numeric_columns]=Df[numeric_columns].apply(pd.to_numeric, errors='coerce')

pf=pd.get_dummies(Df,columns=["X3","X4"])

#drop unique id column
pf.drop(columns=["ID"], inplace=True)

#make correlation matrix for the dataset
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = pf.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

