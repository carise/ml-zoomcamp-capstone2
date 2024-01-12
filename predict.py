import pickle

from flask import Flask, jsonify, request


app = Flask(__name__)


def _load_model(file_path:str):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)


app.dv, app.model = _load_model("model_random_forest.bin")


@app.post("/predict")
def predict():
    req_body = request.json
    X = app.dv.transform(req_body)
    y_pred = app.model.predict_proba(X)[0, 1]
    has_heart_disease = y_pred >= 0.5
    return jsonify({"heart_disease_probability": y_pred, "heart_disease": bool(has_heart_disease)})

