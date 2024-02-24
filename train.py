
import melanoma as mel

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet \
    import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, \
#         EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from tensorflow.keras.applications.resnet_v2 \
    import ResNet50V2, ResNet101V2, ResNet152V2
# from tensorflow.keras.applications.resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, \
    DenseNet201
from tensorflow.keras.applications.nasnet import NASNetMobile, NASNetLarge
# from tensorflow.keras.applications.convnext import ConvNeXtTiny, ConvNeXtSmall, \
#     ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge

import logging
import sys
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

img_height, img_width = utilInstance.getImgSize()

optimizer1 = Adam(lr=0.001)
optimizer2 = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
red_lr= ReduceLROnPlateau(monitor='val_accuracy', patience=3 , verbose=1, factor=0.7)
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 20)

CFG = dict(
			batch_size            =  64,   # 8; 16; 32; 64; bigger batch size => moemry allocation issue
			epochs                =  20,   # 5; 10; 20;
			last_trainable_layers =   0,
			verbose               =   1,   # 0; 1
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
			pretrained_weights = 'imagenet',
			model_optimizer = optimizer2,
			# loss='binary_crossentropy',
			loss='categorical_crossentropy',
			metrics=['accuracy'],
			callbacks = [],

			# Postprocessing
			stopper_patience      =  0,   # 0.01; 0.05; 0.1; 0.2;
			# run_functions_eagerly = False,
            
      # save
      snapshot_path = '/raid/mpsych/MELANOMA/snapshot',
      experiment_noaug = f'{DBname}_noaug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
			experiment_aug = f'{DBname}_aug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
		)
base_model = mel.CNN(CFG=CFG)

classifierDict = {
            mel.NetworkType.ResNet50.name: ResNet50,
            mel.NetworkType.ResNet101.name: ResNet101,
            mel.NetworkType.ResNet152.name: ResNet152,
            mel.NetworkType.Xception.name: Xception,
            mel.NetworkType.InceptionV3.name: InceptionV3,
            mel.NetworkType.VGG16.name: VGG16,
            mel.NetworkType.VGG19.name: VGG19,
            mel.NetworkType.EfficientNetB0.name: EfficientNetB0,
            mel.NetworkType.EfficientNetB1.name: EfficientNetB1,
            mel.NetworkType.EfficientNetB2.name: EfficientNetB2,
            mel.NetworkType.EfficientNetB3.name: EfficientNetB3,
            mel.NetworkType.EfficientNetB4.name: EfficientNetB4,
            mel.NetworkType.EfficientNetB5.name: EfficientNetB5,
            mel.NetworkType.EfficientNetB6.name: EfficientNetB6,
            mel.NetworkType.EfficientNetB7.name: EfficientNetB7,

            mel.NetworkType.ResNet50V2.name: ResNet50V2,
            mel.NetworkType.ResNet101V2.name: ResNet101V2,
            mel.NetworkType.ResNet152V2.name: ResNet152V2,

            mel.NetworkType.MobileNet.name: MobileNet,
            mel.NetworkType.MobileNetV2.name: MobileNetV2,

            mel.NetworkType.DenseNet121.name: DenseNet121,
            mel.NetworkType.DenseNet169.name: DenseNet169,
            mel.NetworkType.DenseNet201.name: DenseNet201,

            mel.NetworkType.NASNetMobile.name: NASNetMobile,
            mel.NetworkType.NASNetLarge.name: NASNetLarge,
        }
DBpreprocessorDict = {
          mel.NetworkType.ResNet50.name: mel.NetworkType.ResNet50.name,
          mel.NetworkType.ResNet101.name: mel.NetworkType.ResNet50.name,
          mel.NetworkType.ResNet152.name: mel.NetworkType.ResNet50.name,
          mel.NetworkType.Xception.name: mel.NetworkType.Xception.name,
          mel.NetworkType.InceptionV3.name: mel.NetworkType.InceptionV3.name,
          mel.NetworkType.VGG16.name: mel.NetworkType.VGG16.name,
          mel.NetworkType.VGG19.name: mel.NetworkType.VGG19.name,
          mel.NetworkType.EfficientNetB0.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB1.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB2.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB3.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB4.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB5.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB6.name: mel.NetworkType.EfficientNetB0.name,
          mel.NetworkType.EfficientNetB7.name: mel.NetworkType.EfficientNetB0.name,

          mel.NetworkType.ResNet50V2.name: mel.NetworkType.ResNet50V2.name,
          mel.NetworkType.ResNet101V2.name: mel.NetworkType.ResNet50V2.name,
          mel.NetworkType.ResNet152V2.name: mel.NetworkType.ResNet50V2.name,

          mel.NetworkType.MobileNet.name: mel.NetworkType.MobileNet.name,
          mel.NetworkType.MobileNetV2.name: mel.NetworkType.MobileNetV2.name,

          mel.NetworkType.DenseNet121.name: mel.NetworkType.DenseNet121.name,
          mel.NetworkType.DenseNet169.name: mel.NetworkType.DenseNet121.name,
          mel.NetworkType.DenseNet201.name: mel.NetworkType.DenseNet121.name,

          mel.NetworkType.NASNetMobile.name: mel.NetworkType.NASNetMobile.name,
          mel.NetworkType.NASNetLarge.name: mel.NetworkType.NASNetMobile.name,
}

# Training DBs with Networks
dbpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001/melanomaDB/customDB/{DBpreprocessorDict[CLASSIFIER]}'
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
model = base_model.transformer(classifierDict[CLASSIFIER])
# model = base_model.inceptionV3()

history_noaug = base_model.fit_model(
    model = model,
    model_name = model_noaug_name,
    trainimages = trainimages,
    trainlabels = trainlabels,
    validationimages = validationimages,
    validationlabels = validationlabels,
)

visualizer = mel.Visualizer()
visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_noaug_name)

visualizer.visualize_performance(
    model_name = model_noaug_name,
    plot_path=CFG['snapshot_path'],
    history = history_noaug
)


dbpath_HAM10000 = dbpath + '/' + f'HAM10000_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_binary.pkl'
dbpath_KaggleMB = dbpath + '/' + f'KaggleMB_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_binary.pkl'
dbpath_ISIC2016 = dbpath + '/' + f'ISIC2016_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_binary.pkl'
dbpath_ISIC2017 = dbpath + '/' + f'ISIC2017_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_binary.pkl'
dbpath_ISIC2018 = dbpath + '/' + f'ISIC2018_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_binary.pkl'
base_model.evaluate_model_onAll(
  model_name=model_noaug_name,
  model_path=CFG['snapshot_path'],
  dbpath_KaggleDB=dbpath_KaggleMB,
  dbpath_HAM10000=dbpath_HAM10000,
  dbpath_ISIC2016=dbpath_ISIC2016,
  dbpath_ISIC2017=dbpath_ISIC2017,
  dbpath_ISIC2018=dbpath_ISIC2018
)

# Augmented images training (augmentation)
model_aug_name = CFG['experiment_aug']

history_aug = base_model.fit_model(
    model = model,
    model_name = model_aug_name,
    trainimages = trainimages_aug,
    trainlabels = trainlabels_aug,
    validationimages = validationimages,
    validationlabels = validationlabels,
)

visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_aug_name)

visualizer.visualize_performance(
    model_name = model_aug_name,
    plot_path=CFG['snapshot_path'],
    history = history_aug
)


base_model.evaluate_model_onAll(
  model_name=model_aug_name,
  model_path=CFG['snapshot_path'],
  dbpath_KaggleDB=dbpath_KaggleMB,
  dbpath_HAM10000=dbpath_HAM10000,
  dbpath_ISIC2016=dbpath_ISIC2016,
  dbpath_ISIC2017=dbpath_ISIC2017,
  dbpath_ISIC2018=dbpath_ISIC2018
)