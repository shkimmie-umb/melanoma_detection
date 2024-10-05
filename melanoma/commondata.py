import melanoma as mel
from enum import Enum



class DatasetType(Enum):
      HAM10000 = 1
      ISIC2016= 2
      ISIC2017=3
      ISIC2018 = 4
      ISIC2019 = 5
      ISIC2020 = 6
      PH2 = 7
      _7_point_criteria = 8
      PAD_UFES_20 = 9
      MEDNODE = 10
      KaggleMB = 11
      

class NetworkType(Enum):
      AlexNet = 1,
      VGG11 = 2,
      VGG13 = 3,
      VGG16 = 4,
      VGG19 = 5,
      VGG11_bn = 6,
      VGG13_bn = 7,
      VGG16_bn = 8,
      VGG19_bn = 9,
      ResNet18 = 10,
      ResNet34 = 11,
      ResNet50 = 12,
      ResNet101 =13,
      ResNet152 = 14,
      SqueezeNet10 = 15,
      SqueezeNet11 = 16,
      Densenet121 = 17,
      Densenet169 = 18,
      Densenet201 = 19,
      Densenet161 = 20
      InceptionV3 = 21,
      GoogleNet = 22,
      ShuffleNetV2x10 = 23,
      ShuffleNetV2x05 = 24,
      MobileNetV2 = 25,
      MobileNetV3Large = 26,
      MobileNetV3Small = 27,
      ResNeXt5032x4d = 28,
      ResNeXt10132x8d = 29,
      WideResNet50_2 = 30,
      WideResNet101_2 = 31,
      MNASNet10 = 32,
      MNASNet05 = 33,
      EfficientNetB1 = 34
      EfficientNetB2 = 35,
      EfficientNetB6 = 36,
      MelaD = 37

    

class CommonData:
      def __init__(self):
         
            self.dbNumImgs = {
            mel.DatasetType.HAM10000: {
                  "trainimages": 10015,
                  "validationimages": 0,
                  "testimages": 0
            },
            mel.DatasetType.ISIC2016: {
                  "trainimages": 900,
                  "trainimgs_benign": 727,
                  "trainimgs_malignant": 173,
                  "validationimages": 0,
                  "testimages": 379,
                  "testimgs_benign": 75,
                  "testimgs_malignant": 304 
                  },
            mel.DatasetType.ISIC2017: {
                  "trainimages": 2000,
                  "trainimgs_benign": 1626,
                  "trainimgs_malignant": 374,
                  "validationimages": 150,
                  "validationimgs_benign": 120,
                  "validationimgs_malignant": 30,
                  "testimages": 600,
                  "testimgs_benign": 483,
                  "testimgs_malignant": 117
                  },
            mel.DatasetType.ISIC2018: {
                  "trainimages": 10015,
                  "trainimgs_benign": 8902,
                  "trainimgs_malignant": 1113,
                  "validationimages": 193,
                  "validationimgs_benign": 172,
                  "validationimgs_malignant": 21,
                  "testimages": 1512,
                  "testimgs_benign": 1341,
                  "testimgs_malignant": 171
                  },
            mel.DatasetType.ISIC2019: {
                  "trainimages": 25331,
                  "trainimgs_benign": 20809,
                  "trainimgs_malignant": 4522,
                  "validationimages": 0,
                  "testimages": 8238 # No ground truth
                  },
            mel.DatasetType.ISIC2020: {
                  "trainimages": 33126,
                  "trainimgs_benign": 32542,
                  "trainimgs_malignant": 584,
                  "validationimages": 0,
                  "testimages": 10982 # No ground truth
                  },
            mel.DatasetType.PH2: {
                  "trainimages": 200,
                  "validationimages": 0,
                  "testimages": 0
                  },
            mel.DatasetType._7_point_criteria: {
                  "trainimages": 413,
                  "trainimgs_benign": 323,
                  "trainimgs_malignant": 90,
                  "validationimages": 203,
                  "validationimgs_benign": 142,
                  "validationimgs_malignant": 61,
                  "testimages": 395,
                  "testimgs_benign": 294,
                  "testimgs_malignant": 101
                  },
            mel.DatasetType.PAD_UFES_20: {
                  "trainimages": 2298,
                  "trainimgs_benign": 2246,
                  "trainimgs_malignant": 52,
                  "validationimages": 0,
                  "testimages": 0
                  },
            mel.DatasetType.KaggleMB: {
            # train: 1440 benign, 1197 malignant; 
            # test: 360 benign + 300 malignant
                  "trainimages": 2637,
                  "trainimgs_benign": 1440,
                  "trainimgs_malignant": 1197,
                  "validationimages": 0,
                  "testimages": 660,
                  "testimgs_benign": 360,
                  "testimgs_malignant": 300
                  },
            mel.DatasetType.MEDNODE: {
            # train: 100 naevus, 70 melanoma
                  "trainimages": 170,
                  "trainimgs_benign": 100,
                  "trainimgs_malignant": 70,
                  "validationimages": 0,
                  "testimages": 0
                  },
            }

      
     