# -*- coding: utf-8 -*-
# @File    : FeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import logging
import numpy as np
from typing import List, Tuple
logging.getLogger(__name__)

from src.BaseClass.BaseFeatureCreator import BaseFeatureCreator


class FeatureCreator(BaseFeatureCreator):
    def __init__(self, **kwargs) -> None:
        super(FeatureCreator, self).__init__()
        self.feature_data = None
        self.feature_cols = None

    def get_seasonality_feature(self):
        self.feature_data['date_time'] = pd.to_datetime(self.feature_data['date_time'], errors='coerce')
        self.feature_data['month'] = self.feature_data['date_time'].dt.month
        self.feature_data['dayofweek'] = self.feature_data['date_time'].dt.dayofweek
        self.feature_cols.extend([
            'month',
            'dayofweek'
        ])

    def get_search_info(self):
        self.feature_data['srch_ci'] = pd.to_datetime(self.feature_data['srch_ci'], errors='coerce')
        self.feature_data['srch_co'] = pd.to_datetime(self.feature_data['srch_co'], errors='coerce')
        self.feature_data['day_distance'] = (self.feature_data['srch_ci'] - self.feature_data['date_time']).dt.days
        self.feature_data['length_of_stay'] = (self.feature_data['srch_co'] - self.feature_data['srch_ci']).dt.days
        def get_is_domestic(row):
            if row['user_location_country'] == row['hotel_country']:
                return 1
            return 0
        self.feature_data['is_domestic'] = self.feature_data.apply(lambda row: get_is_domestic(row), axis=1)
        self.feature_cols.extend([
            'day_distance',
            'length_of_stay',
            'is_domestic'
        ])

    def get_features(self, df,  **kwargs):
        self.feature_data = df
        self.feature_cols = []
        feature_func = [
            self.get_seasonality_feature,
            self.get_search_info
        ]
        for func in feature_func:
            func()
        logging.info(f"features: {self.feature_cols}")
        logging.info(self.feature_data[self.feature_cols].sample(10))
        return self.feature_data
