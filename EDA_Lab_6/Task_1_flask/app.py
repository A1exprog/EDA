from flask import Flask, request, jsonify
import pandas as pd
import json
import joblib
from catboost import CatBoostClassifier

app = Flask(__name__)

# load models
cat_model = CatBoostClassifier()
cat_model.load_model("models/catboost.cbm")

rf_model = joblib.load("models/rf.joblib")

with open("models/features.json") as f:
    FEATURES = json.load(f)

def preprocess(df):
    df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = df[col].astype('category').cat.codes
    df["Weekend"] = df["Weekend"].astype(int)
    return df.reindex(columns=FEATURES, fill_value=0)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_type = data.pop("model", "catboost")

    df = pd.DataFrame([data])
    df = preprocess(df)

    if model_type == "random_forest":
        proba = rf_model.predict_proba(df)[0, 1]
    else:
        proba = cat_model.predict_proba(df)[0, 1]

    label = int(proba >= 0.5)
    return jsonify({
        "model": model_type,
        "Probability of purchase": float(proba),
        "Prediction": label
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
