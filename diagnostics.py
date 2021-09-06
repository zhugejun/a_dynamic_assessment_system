
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from sklearn.metrics import f1_score

with open('config.json', 'r') as f:
    config = json.load(f)

ingested_data_path = os.path.join(config['output_folder_path'])
prod_model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])


def get_test_data():
    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    df.drop('corporation', axis=1, inplace=True)
    y_test = df.pop('exited')
    X_test = df.copy()
    return X_test, y_test


def model_predictions():
    with open(os.path.join(os.getcwd(), prod_model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    X_test, _ = get_test_data()
    y_pred = model.predict(X_test)

    return y_pred


def model_scoring():
    X_test, y_test = get_test_data()
    y_pred = model_predictions()
    return f1_score(y_test, y_pred)


def dataframe_summary():
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(
        os.getcwd(), ingested_data_path, 'finaldata.csv'))
    df.drop('exited', axis=1, inplace=True)
    numerical_columns = [
        col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    df = df[numerical_columns]
    means = df.mean().to_list()
    medians = df.median().to_list()
    stds = df.std().to_list()

    return [means, medians, stds]


def check_missing_percent():
    df = pd.read_csv(os.path.join(
        os.getcwd(), ingested_data_path, 'finaldata.csv'), na_values='NA')
    df.drop('exited', axis=1, inplace=True)

    na_percentages = []

    for col in df.columns:
        percent = df[col].isnull().sum() / len(df)
        na_percentages.append(percent)
    return na_percentages


def execution_time():
    # calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - start_time

    return [ingestion_timing, training_timing]


def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    with open(os.path.join(os.getcwd(), 'outdated_packages.txt'), 'wb') as f:
        f.write(outdated)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    check_missing_percent()
    execution_time()
    outdated_packages_list()
