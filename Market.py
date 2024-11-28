import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('MarketDataset.csv') # Load the dataset
df = df.dropna() # Handle missing values

# Encode categorical features
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Decision', axis=1)
Y = df['Decision']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# Decision Tree Classifier
dt_model = make_pipeline(SimpleImputer(strategy='mean'), DecisionTreeClassifier(criterion='entropy'))
dt_model.fit(X_train, Y_train)
dt_y_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(Y_test, dt_y_pred)
print(f"Decision Tree Accuracy: {dt_acc}")

# Random Forest Classifier
rf_model = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier(n_estimators=100, random_state=0))
rf_model.fit(X_train, Y_train)
rf_y_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_acc}")

# XGBoost Classifier
xgb_model = make_pipeline(SimpleImputer(strategy='mean'), XGBClassifier())
xgb_model.fit(X_train, Y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(Y_test, xgb_y_pred)
print(f"XGBoost Accuracy: {xgb_acc}")

# SVM Classifier
svm_model = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), SVC())
svm_model.fit(X_train, Y_train)
svm_y_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_y_pred)
print(f"SVM Accuracy: {svm_acc}")

# Visualize Decision Tree
dot_graph = export_graphviz(dt_model.steps[1][1], out_file=None, feature_names=X.columns, class_names=df['Decision'].unique().astype(str))
graph = pydotplus.graph_from_dot_data(dot_graph)
graph.write_png("decision_tree.png")
plt.figure(figsize=(15, 20))
plt.imshow(img.imread("decision_tree.png"))
plt.show()
