# -*- coding: utf-8 -*-
# @File    : DummyFeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
from src.BaseClass.BaseFeatureCreator import BaseFeatureCreator


class DummyFeatureCreator(BaseFeatureCreator):
    def __init__(self, **kwargs):
        super(DummyFeatureCreator, self).__init__()

    def get_features(self, df, **kwargs):
        return df

