#!/usr/bin/env python
import os
import time
import numpy as np
from utils import (
    SetSeed,
    Options
)
import polars as pl
import sys
from collections import defaultdict
from dataset_factory import DatasetFactory

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative

if __name__ == "__main__":
    total_start = time.time()
    options = Options(root=ROOT).parse_options()
    options.fix_args()
    datasetfactory = DatasetFactory(args=options.args)()
    options.update_args({'num_classes': datasetfactory.num_classes, 'dataset_path': datasetfactory.path})
    options.display()
    options.save()
    args = options.args
    
    stats = defaultdict(lambda: {'min': [], 'max': []})
    time_per_experiment = []

    for t in range(args.prev, args.times):
        print(f"\n============= Running time: {t}th =============")
        SetSeed(seed=args.seed+t).set()
        print("Creating server and clients ...")
        start = time.time()
        server = getattr(__import__('frameworks'), args.framework)(args, t)
        server.train()
        for key, value in server.metrics.items():
            stats[key]['min'].append(min(value))
            stats[key]['max'].append(max(value))
        ts = time.time()-start
        stats['time_per_experiment']['min'].append(ts)
        stats['time_per_experiment']['max'].append(ts)

    rows = []
    for metric, stats in stats.items():
        row = {
            'metric': metric,
            'avg_min': np.mean(stats['min']),  
            'std_min': np.std(stats['min']),
            'avg_max': np.mean(stats['max']),
            'std_max': np.std(stats['max']),  
        }
        rows.append(row)
    stats = pl.DataFrame(rows)
    stats.write_csv(os.path.join(args.save_path, 'results.csv'))
    print(stats)