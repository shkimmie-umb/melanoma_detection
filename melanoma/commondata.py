import melanoma as mel
from enum import Enum

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
      Multiple = 100
      

class NetworkType(Enum):
      ResNet50 = 1
      ResNet101 = 2
      ResNet152 = 3
      Xception = 4
      InceptionV3 = 5
      VGG16 = 6
      VGG19 = 7
      EfficientNetB0 = 8
      EfficientNetB1 = 9
      EfficientNetB2 = 10
      EfficientNetB3 = 11
      EfficientNetB4 = 12
      EfficientNetB5 = 13
      EfficientNetB6 = 14
      EfficientNetB7 = 15
      # EfficientNetV2B0 = 16
      # EfficientNetV2B1 = 17
      # EfficientNetV2B2 = 18
      # EfficientNetV2B3 = 19
      # EfficientNetV2S = 20
      # EfficientNetV2M = 21
      # EfficientNetV2L = 22
      ResNet50V2 = 23
      ResNet101V2 = 24
      ResNet152V2 = 25
      # InceptionResNetV2 = 26
      MobileNet = 27
      MobileNetV2 = 28
      DenseNet121 = 29
      DenseNet169 = 30
      DenseNet201 = 31
      NASNetMobile = 32
      NASNetLarge = 33
      MeshNet = 34
      # Ensemble = 35
      # ConvNeXtTiny = 34
      # ConvNeXtSmall = 35
      # ConvNeXtBase = 36
      # ConvNeXtLarge = 37
      # ConvNeXtXLarge = 38
      MelaNet = 39

    

class CommonData:
     def __init__(self):
      #    self.classifierDict = {
      #             mel.NetworkType.ResNet50.name: ResNet50,
      #             mel.NetworkType.ResNet101.name: ResNet101,
      #             mel.NetworkType.ResNet152.name: ResNet152,
      #             mel.NetworkType.Xception.name: Xception,
      #             mel.NetworkType.InceptionV3.name: InceptionV3,
      #             mel.NetworkType.VGG16.name: VGG16,
      #             mel.NetworkType.VGG19.name: VGG19,
      #             mel.NetworkType.EfficientNetB0.name: EfficientNetB0,
      #             mel.NetworkType.EfficientNetB1.name: EfficientNetB1,
      #             mel.NetworkType.EfficientNetB2.name: EfficientNetB2,
      #             mel.NetworkType.EfficientNetB3.name: EfficientNetB3,
      #             mel.NetworkType.EfficientNetB4.name: EfficientNetB4,
      #             mel.NetworkType.EfficientNetB5.name: EfficientNetB5,
      #             mel.NetworkType.EfficientNetB6.name: EfficientNetB6,
      #             mel.NetworkType.EfficientNetB7.name: EfficientNetB7,

      #             mel.NetworkType.ResNet50V2.name: ResNet50V2,
      #             mel.NetworkType.ResNet101V2.name: ResNet101V2,
      #             mel.NetworkType.ResNet152V2.name: ResNet152V2,

      #             mel.NetworkType.MobileNet.name: MobileNet,
      #             mel.NetworkType.MobileNetV2.name: MobileNetV2,

      #             mel.NetworkType.DenseNet121.name: DenseNet121,
      #             mel.NetworkType.DenseNet169.name: DenseNet169,
      #             mel.NetworkType.DenseNet201.name: DenseNet201,

      #             mel.NetworkType.NASNetMobile.name: NASNetMobile,
      #             mel.NetworkType.NASNetLarge.name: NASNetLarge,
                  

      # }
         self.dbNumImgs = {
            mel.DatasetType.HAM10000: {
            "trainimages": 10015,
            "validationimages": 0,
            "testimages": 0
            },
            mel.DatasetType.ISIC2016: {
                  "trainimages": 900,
                  "validationimages": 0,
                  "testimages": 379
                  },
            mel.DatasetType.ISIC2017: {
                  "trainimages": 2000,
                  "validationimages": 150,
                  "testimages": 600
                  },
            mel.DatasetType.ISIC2018: {
                  "trainimages": 10015,
                  "validationimages": 193,
                  "testimages": 1512
                  },
            mel.DatasetType.ISIC2019: {
                  "trainimages": 25331,
                  "validationimages": 0,
                  "testimages": 8238 # No ground truth
                  },
            mel.DatasetType.ISIC2020: {
                  "trainimages": 33126,
                  "validationimages": 0,
                  "testimages": 10982
                  },
            mel.DatasetType.PH2: {
                  "trainimages": 200,
                  "validationimages": 0,
                  "testimages": 0
                  },
            mel.DatasetType._7_point_criteria: {
                  "trainimages": 413,
                  "validationimages": 203,
                  "testimages": 395
                  },
            mel.DatasetType.PAD_UFES_20: {
                  "trainimages": 2298,
                  "validationimages": 0,
                  "testimages": 0
                  },
            mel.DatasetType.KaggleMB: {
            # train: 1440 benign, 1197 malignant; 
            # test: 360 benign + 300 malignant
                  "trainimages": 1440+1197,
                  "validationimages": 0,
                  "testimages": 360+300
                  },
            mel.DatasetType.MEDNODE: {
            # train: 70 melanoma, 100 naevus
                  "trainimages": 70+100,
                  "validationimages": 0,
                  "testimages": 0
                  },
            }

         self.DBpreprocessorDict = {
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

                  mel.NetworkType.MeshNet.name: mel.NetworkType.VGG16.name,

                  # mel.NetworkType.Ensemble.name: mel.NetworkType.ResNet50.name,
                  }
      
     @staticmethod
     def model_caller(classifier, CFG):
            classifierDict = {
                  mel.NetworkType.ResNet50.name: mel.CNN.transfer(ResNet50, CFG),
                  mel.NetworkType.ResNet101.name: mel.CNN.transfer(ResNet101, CFG),
                  mel.NetworkType.ResNet152.name: mel.CNN.transfer(ResNet152, CFG),
                  mel.NetworkType.Xception.name: mel.CNN.transfer(Xception, CFG),
                  mel.NetworkType.InceptionV3.name: mel.CNN.transfer(InceptionV3, CFG),
                  mel.NetworkType.VGG16.name: mel.CNN.transfer(VGG16, CFG),
                  mel.NetworkType.VGG19.name: mel.CNN.transfer(VGG19, CFG),
                  mel.NetworkType.EfficientNetB0.name: mel.CNN.transfer(EfficientNetB0, CFG),
                  mel.NetworkType.EfficientNetB1.name: mel.CNN.transfer(EfficientNetB1, CFG),
                  mel.NetworkType.EfficientNetB2.name: mel.CNN.transfer(EfficientNetB2, CFG),
                  mel.NetworkType.EfficientNetB3.name: mel.CNN.transfer(EfficientNetB3, CFG),
                  mel.NetworkType.EfficientNetB4.name: mel.CNN.transfer(EfficientNetB4, CFG),
                  mel.NetworkType.EfficientNetB5.name: mel.CNN.transfer(EfficientNetB5, CFG),
                  mel.NetworkType.EfficientNetB6.name: mel.CNN.transfer(EfficientNetB6, CFG),
                  mel.NetworkType.EfficientNetB7.name: mel.CNN.transfer(EfficientNetB7, CFG),

                  mel.NetworkType.ResNet50V2.name: mel.CNN.transfer(ResNet50V2, CFG),
                  mel.NetworkType.ResNet101V2.name: mel.CNN.transfer(ResNet101V2, CFG),
                  mel.NetworkType.ResNet152V2.name: mel.CNN.transfer(ResNet152V2, CFG),

                  mel.NetworkType.MobileNet.name: mel.CNN.transfer(MobileNet, CFG),
                  mel.NetworkType.MobileNetV2.name: mel.CNN.transfer(MobileNetV2, CFG),

                  mel.NetworkType.DenseNet121.name: mel.CNN.transfer(DenseNet121, CFG),
                  mel.NetworkType.DenseNet169.name: mel.CNN.transfer(DenseNet169, CFG),
                  mel.NetworkType.DenseNet201.name: mel.CNN.transfer(DenseNet201, CFG),

                  mel.NetworkType.NASNetMobile.name: mel.CNN.transfer(NASNetMobile, CFG),
                  mel.NetworkType.NASNetLarge.name: mel.CNN.transfer(NASNetLarge, CFG),
                  mel.NetworkType.MelaNet.name: mel.CNN.melanet(CFG),

                  }
            model = classifierDict[classifier]

            return model