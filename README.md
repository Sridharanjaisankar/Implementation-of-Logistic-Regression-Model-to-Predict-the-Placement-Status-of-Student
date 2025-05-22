# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Load the dataset and check for null data values and duplicate data values in the dataframe.
3. Import label encoder from sklearn.preprocessing to encode the dataset.
4. Apply Logistic Regression on to the model.
5. Predict the y values.
6. Calculate the Accuracy,Confusion and Classsification report.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sridharan J
RegisterNumber:  212222040158
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv') 
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

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
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state =0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## 1.Placement data:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/2a6efe09-4d7c-4dd2-8134-5d5879df5073)

## 2.Salary data:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/47facc95-dfd1-4c65-ae47-cea34e774576)

## 3.Checking the null() function:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/23315b45-7c21-4311-bf53-e61fec7e2b73)

## 4.Data Duplicate:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/7d098d5c-6225-4365-8fe8-9bcee8addd6e)

## 5.Print data:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/e52e47e6-d892-439a-b26a-9f8e1f4385c7)

## 6.Data-status:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/a95eed0d-5921-4412-ab4b-6b58977431b8)


## 7.y_prediction array:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/9730c737-4ba8-4e8b-a4fd-76d606e6d004)

## 8.Accuracy value:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/38decb6b-3bfb-42a8-9f39-800b92c5e8fa)

## 9.Confusion array:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/0cd2ffab-8f1a-4992-84c2-51826a7e857b)

## 10.Classification report:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/bdaa1e28-71f0-424c-9007-f55196929568)

## 11.Prediction of LR:
![image](https://github.com/SOMEASVAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/93434149/1e059472-c2c8-4d56-acfa-f3c882f7a446)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
