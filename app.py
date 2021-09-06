from flask import Flask, session, jsonify, request
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import pickle
import json
import os

import diagnostics


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

def load_prod_model():
    with open(os.path.join(os.getcwd(), prod_model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    return model


@app.route("/prediction")
def predict():
    file_name = request.args.get('file_name')
    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, file_name))
    df.drop('corporation', axis=1, inplace=True)
    y_test = df.pop('exited')
    X_test = df.copy()

    model = load_prod_model()
    y_pred = model.predict(X_test)
    return str(y_pred)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    return str(diagnostics.model_scoring())


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    return  str(diagnostics.dataframe_summary())


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def missing():
    return str(diagnostics.check_missing_percent())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
