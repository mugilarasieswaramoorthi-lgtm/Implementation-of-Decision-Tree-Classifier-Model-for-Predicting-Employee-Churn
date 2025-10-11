# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start and Import the Required Libraries

2.Load and Prepare the Dataset

3.Split the Dataset into Training and Testing Sets

4.Train the Decision Tree Classifier and Make Predictions

5.Evaluate Model Performance and Visualize the Decision Tree 
 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mugilarasi E
RegisterNumber: 25017644

 import pandas as pd 
from sklearn.tree import DecisionTreeClassifier,plot_tree 
data=pd.read_csv("Employee.csv") 
print(data.head()) 
data.info() 
data.isnull().sum() 
data["left"].value_counts() 
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
data["salary"]=le.fit_transform(data["salary"]) 
print(data.head()) 
x = data[["satisfaction_level", 
"last_evaluation", "number_project", 
"average_montly_hours",  
"time_spend_company", 
"Work_accident", "promotion_last_5years", 
"salary"]] 
print(x.head()) 
y=data["left"] 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=
 from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier(criterion="entropy") 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test)
 print("y_pred",y_pred)
 from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y_pred) 
print("accuracy",accuracy) 
import matplotlib.pyplot as plt
 dt.predict([[0.5,0.8,9,260,6,0,1,2]]) 
plt.figure(figsize=(8,6)) 
plot_tree(dt,feature_names=x.columns,class_names=
 ['stayed','left'],filled=True)

```

## Output:
![8th 1](https://github.com/user-attachments/assets/7c8a3fe6-1c74-424d-8cf4-47b031efdc1f)
![8th 2](https://github.com/user-attachments/assets/f156b52e-dbe5-416b-bb49-9d5714ace064)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
