import os
import sys
import time
import numpy as np
from utils import (
    avg_results, 
    SetSeed, 
    Options, 
    DatasetFactory, 
    # MemReporter
)


FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative


if __name__ == "__main__":
    total_start = time.time()
    options = Options(ROOT).parse_options()
    datasetfactory = DatasetFactory(args=options.args)()
    options.add_args('num_classes', datasetfactory.num_classes)
    options.add_args('dataset_path', datasetfactory.path)
    options.fix_args()
    options.display()
    options.save()
    args = options.args
    
    time_list = []

    for t in range(args.times):
        print(f"\n============= Running time: {t}th =============")
        print("Creating server and clients ...")
        start = time.time()
        SetSeed(seed=args.seed+t).set()
        if args.topology is None:
            getattr(__import__('frameworks'), args.framework+'_Server')(configs=args, times=t).train()
        else:
            raise NotImplementedError("This part is not implemented yet.")
            # getattr(__import__('frameworks'), 'TrainingController')(configs=args, times=t).train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    avg_results(path=args.save_path)
    print("All done!")

