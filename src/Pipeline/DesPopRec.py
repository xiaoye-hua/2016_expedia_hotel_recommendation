# -*- coding: utf-8 -*-
# @File    : DesPopRec.py
# @Author  : Hua Guo
# @Disc    :
import os
import pandas as pd
from copy import deepcopy

from src.config import target_col
from src.BaseClass.Pipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval


class DesPopRec(BasePipeline):
    def __init__(self, model_path: str, model_training=True, model_params={}, load_pmml=False, **kwargs) -> None:
        super(DesPopRec, self).__init__(model_training=model_training, model_path=model_path, **kwargs)
        self.pipeline = None
        self.model_params = model_params
        # self.model_file_name = 'pipeline.csv'
        self.des_pop_lst = None
        self.user_historical_booking = None
        self.default_rec = [91, 48, 42, 59, 28]
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        if not self.model_training:
            self.load_pipeline()

    def save_pipeline(self):
        self.des_pop_lst.to_csv(os.path.join(self.model_path, 'des_pop_lst'), index=False)

    def load_pipeline(self):
        self.des_pop_lst = pd.read_csv(os.path.join(self.model_path, 'des_pop_lst'))

    def train(self, X, y, train_params):
        X[target_col] = y
        booked = X[X[target_col]==1]
        clicked = X[X[target_col]!=1]
        pop_booked = booked.groupby('srch_destination_id')['hotel_cluster'].value_counts().to_frame().rename(
            columns={'hotel_cluster': 'num'}).reset_index().groupby('srch_destination_id')['hotel_cluster'].apply(list).reset_index().rename(columns={
            'hotel_cluster': 'hotel_cluster_booked'
        })
        pop_clicked = clicked.groupby('srch_destination_id')['hotel_cluster'].value_counts().to_frame().rename(
            columns={'hotel_cluster': 'num'}).reset_index().groupby('srch_destination_id')['hotel_cluster'].apply(
            list).reset_index().rename(columns={
            'hotel_cluster': 'hotel_cluster_clicked'
        })
        pop_df = pop_clicked.merge(pop_booked, how='left', on='srch_destination_id',
                                   # lsuffix='_l', rsuffix='_r'
                                   )
        def get_pop_lst(row):
            booked = row['hotel_cluster_booked']
            clicked = row['hotel_cluster_clicked']
            res_lst = None
            if not isinstance(booked, list) and pd.isna(booked):
                res_lst = clicked
            elif not isinstance(clicked, list)  and pd.isna(clicked):
                res_lst = booked
            else:
                res_lst = booked + clicked
            default_copy = deepcopy(self.des_pop_lst)
            if len(res_lst) > 5:
                res_lst = res_lst[:5]
            elif len(res_lst)<5:
                res_lst = res_lst + self.default_rec[:5-len(res_lst)]
            return ' '.join([str(ele) for ele in res_lst])
        # pop_df['srch_destination_id'] = pop_df.apply(lambda row: get_des_id(row), axis=1)
        pop_df['hotel_cluster'] = pop_df.apply(lambda row: get_pop_lst(row), axis=1)
        self.des_pop_lst = pop_df[['srch_destination_id', 'hotel_cluster']]

    def predict(self, X):
        X = X.merge(self.des_pop_lst, how='left', on='srch_destination_id')
        X['hotel_cluster'] = X['hotel_cluster'].fillna(' '.join([str(ele) for ele in self.default_rec]))
        return X['hotel_cluster'].values

    def eval(self, **kwargs):
        pass
