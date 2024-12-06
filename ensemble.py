
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import melanoma as mel

import itertools
import glob
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pathlib
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--DB",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="+",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

CLI.add_argument(
  "--CLASSIFIER",
  nargs="?",
  type=str,  # any type/callable can be used here
  
)

args = CLI.parse_args()

check_DBs = [db.name for db in mel.DatasetType]
check_Classifiers = [c.name for c in mel.NetworkType]



DB = args.DB
CLASSIFIER = args.CLASSIFIER


DBname = '+'.join(DB)

networktypes = [c.name for c in mel.NetworkType]

# snapshot_path = os.path.join(pathlib.Path.cwd(), 'snapshot')
snapshot_path = '/project/sansa/melanoma/snapshot/'
# snapshot_path = '/raid/mpsych/MELANOMA/snapshot/'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers/ensemble'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    

mel.Model.extract_positives_per_sample(snapshot_path)
# mel.Model.ensemble(snapshot_path)




print("finish")