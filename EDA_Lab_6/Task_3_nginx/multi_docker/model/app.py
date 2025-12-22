from flask import Flask, request, jsonify
import pandas as pd
import json
import joblib

app = Flask(__name__)

# Load GridSearchCV model
grid_model = joblib.load("explainer.joblib")

# Load feature list
# with open("models/features.json") as f:
#     FEATURES = json.load(f)


def preprocess(df):
    # One-hot для категориальных с drop_first
    df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    
    # Кодируем остальные категориальные колонки
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = df[col].astype('category').cat.codes

    df["Weekend"] = df["Weekend"].astype(int)

    # Упорядочиваем колонки по FEATURES
    return df


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = preprocess(df)

    # Получаем вероятность покупки
    proba = grid_model.predict_proba(df)[0]
    label = int(proba >= 0.5)

    return jsonify({
        "Probability of purchase": float(proba),
        "Prediction": label
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

