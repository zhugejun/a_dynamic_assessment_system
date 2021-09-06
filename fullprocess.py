
import sys
import os

from sklearn.metrics import f1_score
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



with open('config.json', 'r') as f:
    config = json.load(f)

prod_folder_path = config['prod_deployment_path']
source_data_path = config['input_folder_path']
ingested_data_path = config['output_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
logger.info('Reading ingestedfiles.txt')
ingested_file_paths = []
with open(os.path.join(os.getcwd(), prod_folder_path, 'ingestedfiles.txt'), 'r') as file:
    for line in file:
        if '.csv' in line:
            ingested_file_paths.append(line[:-1])

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
logger.info('Checking if there is any new data files.')
proceed = False
for file in os.listdir(source_data_path):
    full_path = os.path.join(os.getcwd(), source_data_path, file)
    if not full_path in ingested_file_paths:
        proceed = True
        ingested_file_paths.append(full_path)


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if proceed:
    logger.info('Ingesting new data files.')
    df = ingestion.merge_multiple_dataframe(ingested_file_paths)
    df.drop('corporation', axis=1, inplace=True)
    y_test = df.pop('exited').values
    X_test = df.copy()
else:
    logger.info('Aborted. No new data files.')
    sys.exit(0)

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
logger.info('Making predictions on new ingested data.')
with open(os.path.join(os.getcwd(), prod_folder_path, 'trainedmodel.pkl'), 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)
y_test = [int(x) for x in y_test]
y_pred = [int(x) for x in y_pred]

logger.info('Scoring')
score = f1_score(y_test, y_pred)


logger.info('Checking model drift.')
model_drift = False
with open(os.path.join(os.getcwd(), prod_folder_path, 'latestscore.txt'), 'r') as f:
    old_score = float(f.readline())
    if score < old_score:
        model_drift = True


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if model_drift:
    logger.info('Model drift. Retraining the model.')
    training.train_model()
    logger.info('Scoring new model.')
    scoring.score_model()
else:
    logger.info('Aborted. No model drift.')
    sys.exit(0)



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
logger.info('Redeploy the new model.')
os.system('python3 deployment.py')



##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logger.info('Running diagnostics and reporting scripts.')
os.system('python3 diagnostics.py')
os.system('python3 reporting.py')





