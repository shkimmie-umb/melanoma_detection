import os
import argparse
import pathlib
import itertools
import glob
import melanoma as mel
from torchvision.transforms import v2
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--DB",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="+",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--IMG_SIZE",
  nargs="+",
  type=int,  # any type/callable can be used here
  action="store"
)
CLI.add_argument(
  "--CLASSIFIER",
  nargs="?",
  type=str,  # any type/callable can be used here
  
)

CLI.add_argument(
  "--JOB_INDEX",
  nargs="?",
  type=str,  # any type/callable can be used here

)

args = CLI.parse_args()

check_DBs = [db.name for db in mel.DatasetType]
check_Classifiers = [c.name for c in mel.NetworkType]

assert set(args.DB).issubset(check_DBs)
assert any(args.CLASSIFIER in item for item in check_Classifiers)

print(f"DB: {args.DB}")
print(f"IMG_SIZE: {args.IMG_SIZE}")
print(f"CLASSIFIER: {args.CLASSIFIER}")
print(f"JOB_INDEX: {args.JOB_INDEX}")

DB = args.DB
IMG_SIZE = tuple(args.IMG_SIZE)
CLASSIFIER = args.CLASSIFIER
JOB_INDEX = args.JOB_INDEX


DBname = '+'.join(DB)

img_size = IMG_SIZE # height, width

# optimizer1 = Adam(learning_rate=1e-5)
# optimizer2 = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
# red_lr= ReduceLROnPlateau(monitor='val_loss', patience=5 , verbose=1, factor=0.8)
# cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)

CFG = dict(
      
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
			# batch_size            =  64,   # 8; 16; 32; 64; bigger batch size => moemry allocation issue
			epochs                =  100,   # 5; 10; 20;

			# Model settings
      num_classes = None,
			pretrained_weights = True,
			criterion = nn.CrossEntropyLoss(),
      optimizer = None,
      scheduler = None,


			# Postprocessing
			stopper_patience      =  0,   # 0.01; 0.05; 0.1; 0.2;
			# run_functions_eagerly = False,
            
      # DB load
      db_path = os.path.join(pathlib.Path.cwd(), 'data', 'melanomaDB'),
      # save
      snapshot_path = os.path.join(pathlib.Path.cwd(), 'snapshot', CLASSIFIER),
      # snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot',
      model_file_name = f'{DBname}_{CLASSIFIER}_{JOB_INDEX}',
			
		)






data_transforms = {
    'Train': v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0)),
        v2.RandomAffine(degrees=90, scale = (0.8, 1.2)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(saturation=(0.7, 1.3), 
                            hue=(-0.1, 0.1),
                            brightness=(0.7, 1.3),
                            contrast=(0.7, 1.3)),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Val': v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test': v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Start training augmented images")

dbs = list(itertools.chain.from_iterable([glob.glob(f'{CFG['db_path']}/{db}/final', recursive=True) for db in DB]))

dataloaders, dataset_sizes = mel.Util.combineDatasets(dbs, preprocessing=data_transforms)
CFG['num_classes'] = len(dataloaders['Train'].dataset.datasets[0].classes)

network = mel.CNN.model_caller(CLASSIFIER)
model_ft = mel.CNN.transfer(network=network, weights=True, CFG=CFG)
model_ft = model_ft.to(CFG['device'])
CFG['optimizer'] = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
CFG['scheduler'] = lr_scheduler.StepLR(CFG['optimizer'], step_size=7, gamma=0.1)

model_ft = mel.Model.train_model(conf=CFG, network=model_ft, data=dataloaders, dataset_sizes=dataset_sizes)