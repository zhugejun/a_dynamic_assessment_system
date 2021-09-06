import pandas as pd
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

COLUMN_NAMES = ['corporation', 'lastmonth_activity',
                'lastyear_activity', 'number_of_employees', 'exited']


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe(list_of_file_paths):

    df = pd.DataFrame(columns=COLUMN_NAMES)

    all_files = []
    for file_path in list_of_file_paths:
        if '.csv' in file_path:
            all_files.append(file_path)
            temp_df = pd.read_csv(file_path)
        else:
            continue

        if set(list(temp_df.columns)) == set(COLUMN_NAMES):
            df = df.append(temp_df).reset_index(drop=True)
        else:
            raise Exception(f'{file_path} has different column names.')

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    df.drop_duplicates(inplace=True)
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    logging.info('Datasets combined.')

    # save file records
    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(datetime.now().strftime('%m/%d/%Y') + '\n')
        for file in all_files:
            f.write(str(file) + '\n')
    logging.info('Records saved.')
    return df

if __name__ == '__main__':

    list_of_file_paths = []
    for file_name in os.listdir('practicedata'):
        if '.csv' in file_name:
            list_of_file_paths.append(os.path.join(os.getcwd(), 'practicedata', file_name))
    _ = merge_multiple_dataframe(list_of_file_paths)
