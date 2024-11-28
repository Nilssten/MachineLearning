import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL.Image import Image
import matplotlib.image as img
import xgboost as xgb

dataset = load_iris()
X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

#model = RandomForestClassifier(criterion="entropy", n_estimators=100)
model = xgb.XGBClassifier(n_estimators=100)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

acc = np.mean((y_pred == Y_test) * 1.0)

print(f"acc: {acc}")
