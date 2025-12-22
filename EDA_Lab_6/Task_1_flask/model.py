# model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('/home/alexus/Desktop/DeepCA/EDA-main/data/X_test.csv')
y = pd.read_csv('/home/alexus/Desktop/DeepCA/EDA-main/data/y_test.csv')

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X, y.values.ravel())

joblib.dump(model, '/home/alexus/Desktop/DeepCA/EDA-main/dashboard/model.pkl')
