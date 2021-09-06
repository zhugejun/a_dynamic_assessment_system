import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


with open('config.json', 'r') as f:
    config = json.load(f)

ingested_data_path = os.path.join(config['output_folder_path'])
prod_model_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])



def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    with open(os.path.join(os.getcwd(), prod_model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    df.drop('corporation', axis=1, inplace=True)
    y_test = df.pop('exited')
    X_test = df.copy()
    metrics.plot_confusion_matrix(model, X_test, y_test)

    plt.savefig(os.path.join(os.getcwd(), output_model_path, 'confusion_matrix2.png'))

if __name__ == '__main__':
    score_model()
