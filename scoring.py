import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
trained_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


def score_model():
    with open(os.path.join(os.getcwd(), trained_model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    df.drop('corporation', axis=1, inplace=True)
    y_test = df.pop('exited')
    X_test = df.copy()

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)

    with open(os.path.join(os.getcwd(), 'latestscore.txt'), 'w') as file:
        file.write(str(score))

    logging.info(f'Score: {score}')

    return score


if __name__ == '__main__':
    score_model()
