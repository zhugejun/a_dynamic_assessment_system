from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

ingested_data_path = os.path.join(config['output_folder_path'])
trained_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_in_production():

    if not os.path.exists(os.path.join(os.getcwd(), prod_deployment_path)):
        os.makedirs(os.path.join(os.getcwd(), prod_deployment_path))

    logging.info('Copying files to production.')
    shutil.copyfile(os.path.join(os.getcwd(), trained_model_path, 'trainedmodel.pkl'), os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'))
    shutil.copyfile(os.path.join(os.getcwd(), ingested_data_path, 'ingestedfiles.txt'), os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt'))
    shutil.copyfile(os.path.join(os.getcwd(), 'latestscore.txt'), os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'))
        
        

if __name__ == '__main__':
    store_model_in_production()