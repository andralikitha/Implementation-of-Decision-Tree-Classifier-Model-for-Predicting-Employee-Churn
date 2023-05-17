# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Andra likitha
RegisterNumber: 212221220006 
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy value:")
accuracy

print("data prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/b5a7f326-ae65-4fe4-8f01-bac7258ac616)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/aae484d3-d52a-4c0a-9c3b-9fd15092efce)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/5857ff5f-72c4-4265-80ba-2f5359877070)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/f71ab96b-4ebd-4c27-80c9-bc094d607218)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/67375816-b020-483d-a66d-ee133bacb4ce)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/c012458c-c989-440e-9a21-190d80bee7b7)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/54968dbd-96c6-46de-abab-3ba58e5a24b0)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131592130/94213645-ccc1-49a9-92a9-73eaa9244375)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
