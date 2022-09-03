# -*- coding: utf-8 -*-
# @File    : create_kaggle_debug_data.py
# @Author  : Hua Guo
# @Disc    :

import pandas as pd
import os
import logging

train_debug_rows = 10000
destination_debug_rows = 1000
test_debug_rows = 1000

production_path = 'data/production/kaggle_original_data'
debug_path = 'data/debug/kaggle_original_data'


all_df = pd.read_csv(os.path.join(production_path, 'train.csv'))
destination_df = pd.read_csv(os.path.join(production_path, 'destinations.csv'))
submission_feature_df = pd.read_csv(os.path.join(production_path, 'test.csv'))

all_df = all_df.sample(train_debug_rows)
destination_df = destination_df.sample(destination_debug_rows)
submission_feature_df = submission_feature_df.sample(test_debug_rows)

submission_df = pd.DataFrame({
    'id': submission_feature_df.index,
    'hotel_cluster': '91 1'
})

all_df.to_csv(os.path.join(debug_path, 'train.csv'), index=False)
destination_df.to_csv(os.path.join(debug_path, 'destinations.csv'), index=False)
submission_feature_df.to_csv(os.path.join(debug_path, 'test.csv'), index=False)
submission_df.to_csv(os.path.join(debug_path, 'sample_submission.csv'), index=False)

