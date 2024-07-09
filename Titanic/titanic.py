import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



data = pd.read_csv('Titanic-Dataset.csv')


data['Age'].fillna(data['Age'].median(), inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data['CabinBool'] = data['Cabin'].notna().astype(int)

data.drop(columns=['Cabin'], inplace=True)

data['Fare'].fillna(data['Fare'].median(), inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

data['Fare'] = np.log1p(data['Fare'])

data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

x = data.drop(columns=['Survived'])
y = data['Survived']

# Séparer les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# noramlisation
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm =  (x_test - np.mean(x_train, axis=0))/np.std(x_train, axis=0)


model = KNeighborsClassifier()
model.fit(x_train_norm,y_train)
y_pred= model.predict(x_test_norm)
model_score = accuracy_score(y_test, y_pred)
print(f"Accuracy score is {model_score}")




