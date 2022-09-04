# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging
from datetime import datetime
import pickle

from scripts.train_config import train_config_detail, result_dir
from src.config import log_dir
    # , train_end_date, train_begin_date, eval_end_date, eval_begin_date, target_threshold
from scripts.train_config import raw_data_path, dir_mark, debug, model_dir
# =============== Config ============

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

res_file_name = '_'.join([dir_mark, 'result.csv'])
final_file = os.path.join(result_dir, res_file_name)

logging.info(f"Basic config:")

logging.info(f"dir_mark: {dir_mark};\nfinal file: {final_file}")



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
destination_latent_include = train_config_detail[dir_mark].get('destination_latent_include', False)


target_col = train_config_detail[dir_mark].get('target_col', None)
feature_used = dense_features + sparse_features

kaggle_original_data_path = os.path.join(raw_data_path, 'kaggle_original_data')

# submission_df = pd.read_csv(os.path.join(kaggle_original_data_path, 'sample_submission.csv'))
# test_df = pd.read_csv(os.path.join(kaggle_original_data_path, 'test.csv'))


# ==================================
# read data from DesPopRec
submission_df = pd.read_csv(os.path.join(result_dir, '0903_PopRec_v1_result.csv'))
test_df = pd.read_csv(os.path.join(kaggle_original_data_path, 'test.csv'))
submission_df['hotel_cluster'] = submission_df.apply(lambda row: [int(ele) for ele in row['hotel_cluster'].split(' ')], axis=1)
submission_df = submission_df.explode('hotel_cluster')
test_df = test_df.merge(submission_df, how='left', on='id')
# +++++++++++++++++++++++++++++++++++


model_path = os.path.join(model_dir, dir_mark)

if feature_clean_func is not None:
    test_df = feature_clean_func(df=test_df)

if destination_latent_include:
    kaggle_original_data_path = os.path.join(raw_data_path, 'kaggle_original_data')
    logging.info(f"Loading destination latent variable from {kaggle_original_data_path}")
    latent_feature = pd.read_csv(os.path.join(kaggle_original_data_path, 'destinations.csv'))
    test_df = test_df.merge(latent_feature, how='left', on='srch_destination_id')

feature_cols = eval_feature_cols = feature_used
assert set(feature_cols)==set(eval_feature_cols), f"Diff: {set(feature_cols)-set(eval_feature_cols)}"
pipeline = pipeline_class(model_path=model_path, model_training=False, model_params={})
feature_creator = feature_creator_class()
feature_df = feature_creator.get_features(df=test_df, train=False)


# ==================================
test_df['prob'] = pipeline.predict(X=feature_df)
test_df = test_df.sort_values(['id', 'prob'], ascending=[True, False])
logging.info(test_df.head(10))
test_df = test_df.groupby('id')['hotel_cluster'].apply(list).reset_index()
test_df['hotel_cluster'] = test_df.apply(lambda row: ' '.join([str(ele) for ele in row['hotel_cluster']]), axis=1)
# +++++++++++++++++++++++++++++++++++
logging.info(test_df.head(10))
test_df.to_csv(final_file, index=False)


# assert submission_df.isna().sum().sum() == 0, f"There null values in the final submission df"
# logging.info(submission_df.sample(10))
# logging.info(f"Saving to {final_file}...")
# submission_df.to_csv(final_file, index=False)


