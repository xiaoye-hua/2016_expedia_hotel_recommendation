# -*- coding: utf-8 -*-
# @File    : run_all.py
# @Author  : Hua Guo
# @Disc    :

import subprocess as sp

cmd_lst = [
    'export PYTHONPATH=./:PYTHONPATH',
    'python scripts/data_cvt.py',
    'python scripts/model_train.py',
    'python scripts/create_submission_result_xgb.py'
]
for cmd in cmd_lst:
    print(cmd)
    sp.run(cmd, shell=True)