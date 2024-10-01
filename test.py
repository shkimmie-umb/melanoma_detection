
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

snapshot_path = os.path.join(pathlib.Path.cwd(), 'snapshot')
# snapshot_path = '/raid/mpsych/MELANOMA/snapshot/'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers/ensemble'

if DB:
    assert set(args.DB).issubset(check_DBs)
    assert any(args.CLASSIFIER in item for item in check_Classifiers)
    modelfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/{CLASSIFIER}/{DBname}_{CLASSIFIER}.pt', recursive=True)]))
    modelnames = list(map(lambda x: pathlib.Path(os.path.basename(x)), modelfiles))
else:
    modelfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*/*.pt', recursive=True)]))
    modelnames = list(map(lambda x: pathlib.Path(os.path.basename(x)), modelfiles))
# modelpath = os.path.abspath(os.path.join(modelfiles, os.pardir))

# modelnames.reverse()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



for idx, model_name in enumerate(modelnames):

    db_path = f'/homes/e35889/sansa/melanoma_detection/data/melanomaDB'

    # netname = [each_m for each_m in networktypes if(each_m in model)]
    # if len(netname) == 1:
    #     model_name = netname[0]
    # elif len(netname) == 2:
    #     model_name = netname[1]
    # else:
    #     assert len(netname) <= 2
    # classifier_name = pathlib.Path(modelfiles[0]).parts[-2]
    classifier_name = pathlib.Path(modelfiles[0]).parent.name
    
    loaded_model = mel.CNN.load_model(model_path=modelfiles[idx], num_classes = 2, device=device)

    mel.Model.evaluate_model_onAll(model=loaded_model, model_path=modelfiles[idx], db_path=db_path, device=device)

    # Save ISIC2020 challenge submission
    # mel.Model.evaluate_leaderboard(model=loaded_model, model_path=modelfiles[idx], db_path=db_path, device=device)
    

mel.Model.extract_performances(snapshot_path)




print("finish")