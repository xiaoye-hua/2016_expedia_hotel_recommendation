# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging
import pickle

from scripts.train_config import train_config_detail #, train_end_date, train_begin_date, eval_end_date, eval_begin_date, target_threshold
from scripts.train_config import raw_data_path, dir_mark, debug, model_dir
    #debug_data_path, raw_data_path,
# from src.config import session_file_name, hotel_file_name
# =============== Config ============

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


# target_col = train_config_detail[dir_mark]['target_col']
pipeline_class = train_config_detail[dir_mark]['pipeline_class']
feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params = {}
# grid_search_dict = train_config_detail[dir_mark].get('grid_search_dict', None)
# model_params = train_config_detail[dir_mark].get('model_params', {})
train_valid = train_config_detail[dir_mark].get('train_valid', False)
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)

target_col = train_config_detail[dir_mark].get('target_col', None)
feature_used = dense_features + sparse_features
# assert feature_used is not None
if not train_config_detail[dir_mark].get('data_dir_mark', False):
    target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
else:
    target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))
logging.info(f"Reading data from {target_raw_data_dir}")

test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))


model_path = os.path.join(model_dir, dir_mark)

if feature_clean_func is not None:
    test_df = feature_clean_func(df=test_df)



logging.info(f"Loading model from {model_path}")
new_pipeline = pipeline_class(model_path=model_path,
                              model_training=False,
                              model_params={}
                              )
logging.info(f"Model eval...")
logging.info(f"There are {len(feature_used)} features:")
logging.info(f"{feature_used}")
new_pipeline.eval(X=test_df[feature_used], y=test_df[target_col],
              performance_result_file='all_data_performance.txt',
              # compare_pmml=True
                  )