# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start

Step 2: Import Required Libraries
Import numpy as np for numerical operations.

Step 3: Load and Explore the Dataset

Step 4: Preprocess the Data
Separate the dataset into:
Feature variables (X): sepal length, sepal width, petal length, petal width.
Target variable (y): species (Setosa, Versicolor, Virginica).

Step 5: Split the Dataset
Split the data into training set and testing set using train_test_split()

Step 6: Create and Configure the SGD Classifier Model
Create an instance of SGDClassifier()

Step 7: Train the SGD Classifier
Fit the SGDClassifier model on the training data using the fit(X_train, y_train) method.

Step 8: Make Predictions
Use the trained model to predict the labels of the test set using predict(X_test).

Step 9: Evaluate the Model
Compare the predicted values with the actual labels (y_test).

Step 10: Visualize the Results (Optional but Recommended)

Step 11: Interpret the Results
Analyze the confusion matrix and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Manisha selvakumari.S.S. 
RegisterNumber: 212223220055
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
print("Name: Manisha selvakumari.S.S.")
print("Reg No: 212223220055")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![Screenshot (259)](https://github.com/user-attachments/assets/38aac60d-9f75-4d1b-972a-f0e5870c12f8)

![Screenshot 2025-04-29 000900](https://github.com/user-attachments/assets/b84709ac-032d-4598-b629-1d441b24da62)

![Screenshot 2025-04-29 000906](https://github.com/user-attachments/assets/c7ebceb0-ef3f-48c7-ade9-4008bd9474e6)

![Screenshot (260)](https://github.com/user-attachments/assets/99a0eae7-ed6b-4cae-9922-1c40f2f879c8)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
