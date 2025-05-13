# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries for model training, data manipulation, and scaling.
2. Separate features (X) and target variables (y) for price and occupants.
3. Split data into training and testing sets for both targets.
4. Standardize features using StandardScaler for better performance.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Hari Nivedhan P
RegisterNumber: 212224220031
  
*/

import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data.csv")
print(data.head())
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print(data1)
x=data1.iloc[:,:-1]
print(x)
y=data1["status"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion) 
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
![Screenshot 2025-05-06 083340](https://github.com/user-attachments/assets/62229881-9419-423d-ad8a-81a592b28d8a)

![Screenshot 2025-05-06 083408](https://github.com/user-attachments/assets/1adab10d-01b1-463f-b3a7-4d05fd6c7785)
![Screenshot 2025-05-06 083426](https://github.com/user-attachments/assets/5f03e666-244a-449c-9c97-425e3c89c712)
![Screenshot 2025-05-06 083443](https://github.com/user-attachments/assets/0cf9cc5e-e3f4-457f-842c-448decaa5d14)
![Screenshot 2025-05-06 083459](https://github.com/user-attachments/assets/153e6fdb-4c22-46b4-ad69-df658d107af2)
