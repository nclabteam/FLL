#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from utils import (
    SetSeed,
    Options,
    DatasetFactory
)
import polars as pl
import sys

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative

# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

if __name__ == "__main__":
    total_start = time.time()
    options = Options(root=ROOT).parse_options()
    options.fix_args()
    datasetfactory = DatasetFactory(args=options.args)()
    options.update_args({'num_classes': datasetfactory.num_classes, 'dataset_path': datasetfactory.path})
    options.display()
    options.save()
    args = options.args
    
    time_list = []
    test_personal_accs = []
    test_global_accs = []

    for t in range(args.prev, args.times):
        print(f"\n============= Running time: {t}th =============")
        SetSeed(seed=args.seed+t).set()
        start = time.time()
        if args.topology is None:
            print("Creating server and clients ...")
            server = getattr(__import__('frameworks'), args.framework)(args, t)
            server.train()
            test_personal_accs.append(max(server.metrics['test_personal_accs']))
            test_global_accs.append(max(server.metrics['test_global_accs']))
        else:
            tmanager = getattr(__import__('frameworks'), 'TrainingController')(configs=args, times=t)
            tmanager.train()
            test_personal_accs.append(max(tmanager.metrics['test_personal_accs']))
            test_global_accs.append(0)
        time_list.append(time.time()-start)

    stats_df = pl.DataFrame({
        'Metric': [
            'Mean of max personal accs', 
            'Std of max personal accs',
            'Mean of max global accs', 
            'Std of max global accs',
            'Mean of time per experiments'
        ],
        'Value': [
            np.mean(test_personal_accs), 
            np.std(test_personal_accs), 
            np.mean(test_global_accs), 
            np.std(test_global_accs), 
            np.mean(time_list)
        ]
    })
    stats_df.write_csv(os.path.join(args.save_path, 'results.csv'))
    print(stats_df)