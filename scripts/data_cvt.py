# -*- coding: utf-8 -*-
# @File    : reg_data_cvt.py
# @Author  : Hua Guo
# @Disc    :

import pandas as pd
import os
import logging
from datetime import datetime

from scripts.train_config import train_config_detail #, train_end_date, train_begin_date, eval_end_date, eval_begin_date, test_begin_date, test_end_date
from scripts.train_config import raw_data_path, dir_mark, debug #, debug_date, offer_served_data_source
from src.utils import check_create_dir
from src.config import log_dir
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params = {}
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_used = dense_features + sparse_features
target_col = train_config_detail[dir_mark]['target_col']
# logging.info(f"training date: {train_begin_date}, {train_end_date}")
# logging.info(f"eval date: {eval_begin_date}, {eval_end_date}")
# logging.info(f"test date: {test_begin_date}, {test_end_date}")

curDT = datetime.now()
date_time = curDT.strftime("%Y%m%d%H")
current_file = os.path.basename(__file__).split('.')[0]
log_file = '_'.join([current_file, date_time, '.log'])
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(log_dir, log_file)
                    )
console = logging.StreamHandler()
logging.getLogger().addHandler(console)


train_end_time = '2014-03-01 00:00:00'
train_eval_end_time = '2014-07-01 00:00:00'

# train_end_time = '2014-12-31 23:30:00'
# train_eval_end_time = '2014-12-31 23:45:00'

kaggle_original_data_path = os.path.join(raw_data_path, 'kaggle_original_data')
model_path = os.path.join('model_training/', dir_mark)

logging.info(f"Reading data from {raw_data_path}")

# feature_creator =


data = pd.read_csv(os.path.join(kaggle_original_data_path, 'train.csv'))
logging.info(data.columns)
logging.info(data.info())
train = data[data['date_time']<train_end_time]
eval = data[(data['date_time']>=train_end_time) & (data['date_time']<train_eval_end_time)]
test = data[data['date_time']>=train_eval_end_time]
target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
check_create_dir(directory=target_raw_data_dir)

fc = feature_creator_class()
# feature_creator
import time
begin = time.time()
train_df = fc.get_features(df=train)
end = time.time()
logging.info(f"Train data time: {round((end-begin)/3600, 3)} hours")
# if not debug:X
begin = time.time()
eval_df = fc.get_features(df=eval)
end = time.time()
logging.info(f"Eval data time: {round((end-begin)/3600, 3)} hours")
# if not debug:X
begin = time.time()
test_df = fc.get_features(df=test)
end = time.time()
logging.info(f"Test data time: {round((end-begin)/3600, 3)} hours")

logging.info(f"Saving data to dir: {target_raw_data_dir}")
logging.info(f"train_df {train_df.shape}\nEval_df: {eval_df.shape}\nTest_df: {test_df.shape}")
train_df.to_csv(os.path.join(target_raw_data_dir, 'train.csv'), index=False)
eval_df.to_csv(os.path.join(target_raw_data_dir, 'eval.csv'), index=False)
test_df.to_csv(os.path.join(target_raw_data_dir, 'test.csv'), index=False)




