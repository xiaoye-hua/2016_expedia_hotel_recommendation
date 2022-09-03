# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Disc    :

from src.Pipeline.DesPopRec import DesPopRec
from src.Pipeline.XGBClassifierPipeline import XGBClassifierPipeline
from src.FeatureCreator.DummyFeatureCreator import DummyFeatureCreator
from src.config import target_col

debug = True
dir_mark = "0903_xgb_v1"

if debug:
    raw_data_path = 'data/debug'
    model_dir = 'model_training/debug'
    result_dir = 'result/debug'
    mlflow_cate = 'debug'
else:
    raw_data_path = 'data/production'
    model_dir = 'model_training/'
    result_dir = 'result/production'
    mlflow_cate = 'production'

train_config_detail = {
    "0903_PopRec_v1": {
        'pipeline_class': DesPopRec
        , 'feature_creator': DummyFeatureCreator
        , 'train_valid': True
        , 'sparse_features': [
        ]
        , 'dense_features': [
        ]
        # , 'feature_clean_func': clean_map_feature
        , 'target_col': target_col
    },
    "0903_xgb_v1": {
        'pipeline_class': XGBClassifierPipeline
        , 'feature_creator': DummyFeatureCreator
        , 'train_valid': True
        , 'sparse_features': [
            'srch_destination_type_id'
        ]
        , 'dense_features': [
        ]
        , 'onehot': [
            'srch_destination_type_id'
        ]
        # , 'feature_clean_func': clean_map_feature
        , 'target_col': target_col
    },
}
