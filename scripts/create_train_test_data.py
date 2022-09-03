# # -*- coding: utf-8 -*-
# # @File    : create_kaggle_debug_data.py
# # @Author  : Hua Guo
# # @Disc    :
#
# import pandas as pd
# import os
# import logging
#
#
#
# train_eval_end_time = '2014-07-01 00:00:00'
# debug_data_dir = 'data/debug'
#
# all_df = pd.read_csv(os.path.join(kaggle_original_data_path, 'train.csv'))
# destination_df = pd.read_csv('data/production/destinations.csv')
# submission_feature_df = pd.read_csv('data/production/test.csv')
#
# train_eval_df = all_df[all_df['datetime']<train_eval_end_time]
# test_df = all_df[all_df['datetime']>=train_eval_end_time]
#
# submission_df = pd.DataFrame({
#     'id': submission_feature_df.index,
#     'hotel_cluster': '91 1'
# })
#
# all_df.to_csv(os.path.join(debug_data_dir, 'train.csv'), index=False)
# destination_df.to_csv(os.path.join(debug_data_dir, 'destinations.csv'), index=False)
# submission_feature_df.to_csv(os.path.join(debug_data_dir, 'test.csv'), index=False)
# submission_df.to_csv(os.path.join(debug_data_dir, 'sample_submission.csv'), index=False)
#
