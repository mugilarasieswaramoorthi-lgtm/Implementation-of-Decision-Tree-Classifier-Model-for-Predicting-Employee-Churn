# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mugilarasi E
RegisterNumber: 25017644

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 30, 28, 40, 35, 29, 50, 45, 32, 27, 38, 42],
    'Experience': [2, 5, 3, 15, 10, 4, 25, 20, 6, 3, 12, 18],
    'Department': [1, 2, 2, 3, 1, 2, 3, 4, 2, 1, 4, 3],
    'Education_Level': [1, 2, 2, 3, 1, 2, 3, 2, 1, 1, 2, 3],
    'Salary': [35, 50, 45, 80, 60, 48, 120, 90, 55, 40, 75, 85],
    'Work_Environment_Score': [7, 8, 6, 5, 7, 8, 4, 6, 7, 7, 5, 6],
    'Churn': [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1] 
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

X = df.drop('Churn', axis=1).values
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

plt.figure(figsize=(15,10))
plot_tree(classifier, feature_names=df.columns[:-1], class_names=['Stayed','Left'], filled=True)
plt.show()

new_employee = np.array([[29, 5, 2, 2, 50, 8]]) 
churn_prediction = classifier.predict(new_employee)
print("Churn Prediction for new employee:", "Left" if churn_prediction[0]==1 else "Stayed")

```

## Output:
<img width="537" height="584" alt="Screenshot 2025-10-06 212151" src="https://github.com/user-attachments/assets/cf9acfc9-5462-4950-8e45-f7235b471183" />
<img width="986" height="623" alt="Screenshot 2025-10-06 212209" src="https://github.com/user-attachments/assets/cf4d578b-7954-4973-8790-b2ec59f6cc05" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
