
import melanoma as mel


import logging
import sys
import os
from pathlib import Path
import itertools
import glob
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

import argparse
# defined command line options
# this also generates --help and error handling
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

# parse the command line
args = CLI.parse_args()
# access CLI options

check_DBs = [db.name for db in mel.DatasetType]
check_Classifiers = [c.name for c in mel.NetworkType]

assert set(args.DB).issubset(check_DBs)
assert any(args.CLASSIFIER in item for item in check_Classifiers)

print(f"DB: {args.DB}")
print(f"IMG_SIZE: {args.IMG_SIZE}")
print(f"CLASSIFIER: {args.CLASSIFIER}")
print(f"JOB_INDEX: {args.JOB_INDEX}")
# print("DB: %r" % args.DB)
# print("IMG_SIZE: %r" % args.IMG_SIZE)
# print("CLASSIFIER: %r" % args.CLASSIFIER)
# print("JOB_INDEX: %r" % args.JOB_INDEX)

DB = args.DB
IMG_SIZE = tuple(args.IMG_SIZE)
CLASSIFIER = args.CLASSIFIER
JOB_INDEX = args.JOB_INDEX
# DB = sys.argv[1] # HAM10000, ISIC2016
# IMG_SIZE = int(sys.argv[2]) # (150, 150)
# CLASSIFIER = sys.argv[3] # 'ResNet50, VGG16'
# JOB_INDEX = int(sys.argv[4])

DBname = '+'.join(DB)

rootpath = '/hpcstor6/scratch01/s/sanghyuk.kim001'
# img_size = (224, 224) # height, width
img_size = IMG_SIZE # height, width
utilInstance = mel.Util(rootpath, img_size)

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import EarlyStopping
# from mel import SilentTrainingCallback as silent_callback

img_height, img_width = utilInstance.getImgSize()

optimizer1 = Adam(learning_rate=0.001)
optimizer2 = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
red_lr= ReduceLROnPlateau(monitor='val_accuracy', patience=3 , verbose=1, factor=0.7)
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 20)

CFG = dict(
			batch_size            =  32,   # 8; 16; 32; 64; bigger batch size => moemry allocation issue
			epochs                =  3,   # 5; 10; 20;
			last_trainable_layers =   0,
			verbose               =   0,   # 0; 1
			fontsize              =  14,
			num_classes           =  2, # binary

			# Images sizes
			img_height = img_height,   # Original: (450h, 600w)
            img_width = img_width,

			# Images augs
			ROTATION_RANGE        =   90.0,
			ZOOM_RANGE            =   0.1,
			HSHIFT_RANGE          =   0.1, # randomly shift images horizontally
			WSHIFT_RANGE          =   0.1, # randomly shift images vertically
			HFLIP                 = False, # randomly flip images
			VFLIP                 = False, # randomly flip images

			# Model settings
			# pretrained_weights = 'imagenet',
      pretrained_weights = None,
			model_optimizer = optimizer2,
			# loss='binary_crossentropy',
			loss='categorical_crossentropy',
			metrics=['accuracy'],
			callbacks = [mel.SilentTrainingCallback()],

			# Postprocessing
			stopper_patience      =  0,   # 0.01; 0.05; 0.1; 0.2;
			# run_functions_eagerly = False,
            
      # save
      # snapshot_path = '/raid/mpsych/MELANOMA/snapshot',
      snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot',
      experiment_noaug = f'{DBname}_noaug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
			experiment_aug = f'{DBname}_aug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
		)
base_model = mel.CNN(CFG=CFG)
commondata = mel.CommonData()


# Training DBs with Networks
dbpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001/melanomaDB/customDB/{commondata.DBpreprocessorDict[CLASSIFIER]}'
# picklename = f'{DB}_{IMG_SIZE}h_{IMG_SIZE}w_binary'
del_augmentation = {'ROTATION_RANGE':0.0, 'ZOOM_RANGE':0.0, 'HSHIFT_RANGE':0.0, 'WSHIFT_RANGE':0.0}
CFG.update(del_augmentation)

ori_pkl = list(itertools.chain.from_iterable([glob.glob(f'{dbpath}/{db}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w*', recursive=True) for db in DB]))
aug_pkl = list(itertools.chain.from_iterable([glob.glob(f'{dbpath}/{db}_augmentedWith_*Melanoma_*Non-Melanoma_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w*', recursive=True) for db in DB]))

assert len(ori_pkl) == len(aug_pkl)


	# trainimages, testimages, validationimages, \
	# 			trainlabels, testlabels, validationlabels, num_classes\
	# 				= utilInstance.loadDatasetFromFile(dbpath+'/'+Path(ori_pkl).stem+'.pkl')
trainimages, testimages, validationimages, \
      trainlabels, testlabels, validationlabels, \
        = utilInstance.combineDatasets(ori_pkl)

trainimages_aug, testimages_aug, validationimages_aug, \
      trainlabels_aug, testlabels_aug, validationlabels_aug, \
        = utilInstance.combineDatasets(aug_pkl)

assert len(trainimages) == len(trainlabels)
if validationimages is not None and validationlabels is not None:
  assert len(validationimages) == len(validationlabels)
  assert len(validationimages) == len(validationimages_aug)
  assert len(validationlabels) == len(validationlabels_aug)
if testimages is not None and testlabels is not None:
  assert len(testimages) == len(testlabels)
  assert len(testimages) == len(testimages_aug)
  assert len(testlabels) == len(testlabels_aug)
assert len(trainimages_aug) == len(trainlabels_aug)
if validationimages_aug is not None and validationlabels_aug is not None:
  assert len(validationimages_aug) == len(validationlabels_aug)
if testimages_aug is not None and testlabels_aug is not None:
  assert len(testimages_aug) == len(testlabels_aug)

# Test, Val sets must not be augmented


# Original images training (No augmentation)
model_noaug_name = CFG['experiment_noaug']
model = base_model.transformer(commondata.classifierDict[CLASSIFIER])

modelfiles = list(itertools.chain.from_iterable([glob.glob(f"{CFG['snapshot_path']}/*.hdf5", recursive=True)]))
modelnames = list(map(lambda x: Path(os.path.basename(x)).stem, modelfiles))


history_noaug = base_model.fit_model(
    model = model,
    model_name = model_noaug_name,
    trainimages = trainimages,
    trainlabels = trainlabels,
    validationimages = validationimages,
    validationlabels = validationlabels,
)

  # visualizer = mel.Visualizer()
  # visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_noaug_name)

  # visualizer.visualize_performance(
  #     model_name = model_noaug_name,
  #     plot_path=CFG['snapshot_path'],
  #     history = history_noaug
  # )




# Augmented images training (augmentation)
model_aug_name = CFG['experiment_aug']

base_model.CFG.update({'callbacks': [mel.SilentTrainingCallback()]})


history_aug = base_model.fit_model(
    model = model,
    model_name = model_aug_name,
    trainimages = trainimages_aug,
    trainlabels = trainlabels_aug,
    validationimages = validationimages,
    validationlabels = validationlabels,
)

  # visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_aug_name)

  # visualizer.visualize_performance(
  #     model_name = model_aug_name,
  #     plot_path=CFG['snapshot_path'],
  #     history = history_aug
  # )

# if not any(model_aug_name in has_model for has_model in modelnames):

# else:
#   print(f"Model {model_noaug_name} already exists")