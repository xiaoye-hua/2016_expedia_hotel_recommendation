# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Disc    :

from src.Pipeline.DesPopRec import DesPopRec
from src.Pipeline.XGBClassifierPipeline import XGBClassifierPipeline
from src.FeatureCreator.DummyFeatureCreator import DummyFeatureCreator
from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.config import target_col

debug = False
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


srch_destination_id_feature_lst = ['d'+str(ele) for ele in range(1, 150)]

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
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'destination_latent_include': False
        , 'sparse_features': [
            'srch_destination_type_id',
            'posa_continent',
            'is_mobile',
            'is_package',
            'channel',
            'hotel_continent',
        'hotel_market',
        'hotel_cluster',
        'month', 'dayofweek','is_domestic'
        ]
        , 'dense_features': [
           'orig_destination_distance',
            'srch_adults_cnt',
            'srch_children_cnt',
            'srch_rm_cnt',
            'day_distance',
            'length_of_stay',
        ] # + srch_destination_id_feature_lst
        , 'onehot': [
            'srch_destination_type_id',
            'posa_continent',
            'channel',
            'srch_destination_type_id',
            'hotel_continent',
        'hotel_market',
        'hotel_cluster',
        'month', 'dayofweek',
        ]
        # , 'feature_clean_func': clean_map_feature
        , 'target_col': target_col
    },
}