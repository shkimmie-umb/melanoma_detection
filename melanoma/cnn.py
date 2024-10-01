# import Model
from .model import Model as Base_Model
import melanoma as mel
import torch
import torch.nn as nn

import os
import pathlib
import copy
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg11, vgg11_bn
from torchvision.models import vgg13, vgg13_bn
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import vgg19, vgg19_bn
from torchvision.models import wide_resnet50_2
from torchvision.models import alexnet
from torchvision.models import squeezenet1_0, squeezenet1_1
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import inception_v3
from torchvision.models import googlenet
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import resnext50_32x4d, resnext101_32x8d
from torchvision.models import wide_resnet50_2, wide_resnet101_2
from torchvision.models import mnasnet0_5, mnasnet1_0



class CNN(Base_Model):
    
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def modifyOutputLayer(model_ft, model_name, num_classes):
        if (model_name in (mel.NetworkType.ResNet18.name, mel.NetworkType.ResNet34.name,
        mel.NetworkType.ResNet50.name, mel.NetworkType.ResNet101.name, mel.NetworkType.ResNet152.name,
        mel.NetworkType.InceptionV3.name, mel.NetworkType.GoogleNet.name,
        mel.NetworkType.ShuffleNetV2x05.name, mel.NetworkType.ShuffleNetV2x10.name,
        mel.NetworkType.ResNeXt5032x4d.name, mel.NetworkType.ResNeXt10132x8d.name,
        mel.NetworkType.WideResNet50_2.name, mel.NetworkType.WideResNet101_2.name)):
            num_ftrs = model_ft.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # model_ft.fc = nn.Sequential(
            #     nn.Linear(num_ftrs, 256),  # Additional linear layer with 256 output features
            #     nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            #     nn.Dropout(0.5),               # Dropout layer with 50% probability
            #     nn.Linear(256, num_classes)    # Final prediction fc layer
            # )
        elif (model_name in 
        (mel.NetworkType.VGG11.name, mel.NetworkType.VGG11_bn.name,
        mel.NetworkType.VGG13.name, mel.NetworkType.VGG13_bn.name,
        mel.NetworkType.VGG16.name, mel.NetworkType.VGG16_bn.name,
        mel.NetworkType.VGG19.name, mel.NetworkType.VGG19_bn.name, mel.NetworkType.AlexNet.name,
        mel.NetworkType.MobileNetV2.name, mel.NetworkType.MobileNetV3Large.name, mel.NetworkType.MobileNetV3Small.name,
        mel.NetworkType.MNASNet10.name, mel.NetworkType.MNASNet05.name)):
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        elif (model_name in 
        (mel.NetworkType.SqueezeNet10.name, mel.NetworkType.SqueezeNet11.name)):
            num_ftrs = model_ft.classifier[1].in_channels
            kernel_size = model_ft.classifier[1].kernel_size
            stride = model_ft.classifier[1].stride
            model_ft.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=kernel_size, stride=stride)
        elif (model_name in 
        (mel.NetworkType.Densenet121.name, mel.NetworkType.Densenet161.name,
        mel.NetworkType.Densenet169.name, mel.NetworkType.Densenet201.name)):
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            raise AssertionError("Unknown network")

        return model_ft
    @staticmethod
    def load_model(model_path, num_classes, device):
        classifier_name = pathlib.Path(model_path).parent.name
        network = mel.CNN.model_caller(classifier=classifier_name)
        model_ft = network(weights=True)
        model_ft = mel.CNN.modifyOutputLayer(model_ft=model_ft, model_name=classifier_name, num_classes=num_classes)

        state_dict = torch.load(model_path, map_location=device)
        
        if (list(state_dict.keys())[0][:6] == 'module'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            if(hasattr(state_dict, '_metadata')):
                new_state_dict._metadata = copy.deepcopy(state_dict._metadata)
                for k, v in state_dict.items():
                    name = k[7:] # remove 'module.' of dataparallel
                    new_state_dict[name]=v
            else:
                for k, v in state_dict.items():
                    name = k[7:] # remove 'module.' of dataparallel
                    new_state_dict[name]=v
            
        
            model_ft.load_state_dict(new_state_dict)
        else:
            model_ft.load_state_dict(state_dict)


        model_ft.eval()
        
        return model_ft

    @staticmethod
    def model_caller(classifier):
        classifierDict = {
            mel.NetworkType.AlexNet.name: alexnet,
            mel.NetworkType.VGG11.name: vgg11,
            mel.NetworkType.VGG13.name: vgg13,
            mel.NetworkType.VGG16.name: vgg16,
            mel.NetworkType.VGG19.name: vgg19,
            mel.NetworkType.VGG11_bn.name: vgg11_bn,
            mel.NetworkType.VGG13_bn.name: vgg13_bn,
            mel.NetworkType.VGG16_bn.name: vgg16_bn,
            mel.NetworkType.VGG19_bn.name: vgg19_bn,
            mel.NetworkType.ResNet18.name: resnet18,
            mel.NetworkType.ResNet34.name: resnet34,
            mel.NetworkType.ResNet50.name: resnet50,
            mel.NetworkType.ResNet101.name: resnet101,
            mel.NetworkType.ResNet152.name: resnet152,
            mel.NetworkType.SqueezeNet10.name: squeezenet1_0,
            mel.NetworkType.SqueezeNet11.name: squeezenet1_1,
            mel.NetworkType.Densenet121.name: densenet121,
            mel.NetworkType.Densenet161.name: densenet161,
            mel.NetworkType.Densenet169.name: densenet169,
            mel.NetworkType.Densenet201.name: densenet201,
            mel.NetworkType.InceptionV3.name: inception_v3,
            mel.NetworkType.GoogleNet.name: googlenet,
            mel.NetworkType.ShuffleNetV2x05.name: shufflenet_v2_x0_5,
            mel.NetworkType.ShuffleNetV2x10.name: shufflenet_v2_x1_0,
            mel.NetworkType.MobileNetV2.name: mobilenet_v2,
            mel.NetworkType.MobileNetV3Large.name: mobilenet_v3_large,
            mel.NetworkType.MobileNetV3Small.name: mobilenet_v3_small,
            mel.NetworkType.ResNeXt5032x4d.name: resnext50_32x4d,
            mel.NetworkType.ResNeXt10132x8d.name: resnext101_32x8d,
            mel.NetworkType.WideResNet50_2.name: wide_resnet50_2,
            mel.NetworkType.WideResNet101_2.name: wide_resnet101_2,
            mel.NetworkType.MNASNet05.name: mnasnet0_5,
            mel.NetworkType.MNASNet10.name: mnasnet1_0,
        }
        model = classifierDict[classifier]

        return model

    @staticmethod
    def transfer(network, network_name, weights, CFG):
        
        model_ft = network(weights=weights)

        # for param in model_ft.parameters():
        #     param.requires_grad = False

        # Freeze only the convolutional layers of the pre-trained model
        for param in model_ft.parameters():
            if isinstance(param, nn.Conv2d):
                param.requires_grad = False
        
        model_ft = mel.CNN.modifyOutputLayer(model_ft=model_ft, model_name=network_name, num_classes=CFG['num_classes'])

        

        return model_ft

    @staticmethod
    def melanet(CFG):
        # Define model with different applications
        model = Sequential()
        model.add(Input(shape=(CFG['img_height'], CFG['img_width'], 3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(layers.Conv2D(32,(7,7),padding='same',activation='relu', strides=4))
        model.add(layers.Conv2D(32,(7,7),padding='valid',activation='relu', strides=1))
        model.add(layers.Conv2D(32,(7,7),padding='valid',activation='relu', strides=1))
        model.add(layers.MaxPooling2D((2, 2), strides=None, name='block1_pool'))
        model.add(layers.Conv2D(64,(13,13),padding='valid',activation='relu', strides=1))
        model.add(layers.MaxPooling2D((2, 2), strides=None, name='block2_pool'))
        model.add(Dense(2304, activation=None))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))

        model.summary()

        model.compile(optimizer=CFG['model_optimizer'], loss=CFG['loss'], metrics=CFG['metrics'])
       
        return model

    

    def meshnet_pretrained(self, network=None):
        model = Sequential()
        model.add(network(
                include_top=False,
                input_tensor=None,
                input_shape=(self.CFG['img_height'], self.CFG['img_width'], 3),
                pooling='avg',
                # classes=self.CFG['num_classes'],
                weights=self.CFG['pretrained_weights']
        ))

        model.add(tf.keras.layers.Reshape((150,150,3)))
        
        # Layer 1: Convolutional + Relu + BatchNorm + Dropout
        
        model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        # Layers 2-6: Repetition of Convolutional + Relu + BatchNorm + Dropout with increasing dilation rates
        """
        Our architecture entirely consists of what [7] used as a context module but we modified it to use 3D dilated convolutions.
        https://arxiv.org/pdf/1612.00940.pdf (MeshNet Section)
        """
        dilation_rates = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
        for rate in dilation_rates:
            model.add(Conv2D(filters=64, kernel_size=(3, 3),
                    padding='same', dilation_rate=rate))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))

        # Layer 7: Convolutional + BatchNorm + Activation + Dropout
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Current Pipeline layers
        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(1, activation='sigmoid'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))

        model.add(GlobalAveragePooling2D())

        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))

        

        model.add(Dense(2, activation='softmax'))

        # model.layers[0].trainable = False

        model.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        model.summary()

        return model

    def meshnet(self, network=None):
        # base_model = ResNet50(include_top=False, input_shape=(
        #     self.CFG['img_height'], self.CFG['img_width'], 3), pooling='avg', weights=self.CFG['pretrained_weights'])

        # Define model with different applications
        model = Sequential()
        model.add(Input(shape=(150, 150, 3)))
        # model.add(base_model)
        
        # image = Input(shape=(150, 150, 3))
        # x = Conv2D(3, kernel_size=(3,3), padding='same', activation='relu')

        # MeshNet-inspired layers adapted for 2D, including dilation
        # model.add(Dense(512, activation='relu'))

        # model.add(layers.Conv2D(2048,(3,3),padding='same',activation='relu'))
        # model.add(layers.Conv2D(1024,(3,3),padding='same',activation='relu'))
        # model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
        # Layer 1: Convolutional + Relu + BatchNorm + Dropout
        
        model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        # Layers 2-6: Repetition of Convolutional + Relu + BatchNorm + Dropout with increasing dilation rates
        """
        Our architecture entirely consists of what [7] used as a context module but we modified it to use 3D dilated convolutions.
        https://arxiv.org/pdf/1612.00940.pdf (MeshNet Section)
        """
        dilation_rates = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
        for rate in dilation_rates:
            model.add(Conv2D(filters=64, kernel_size=(3, 3),
                    padding='same', dilation_rate=rate))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))

        # Layer 7: Convolutional + BatchNorm + Activation + Dropout
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Current Pipeline layers
        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(1, activation='sigmoid'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))

        model.add(GlobalAveragePooling2D())

        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))

        

        model.add(Dense(2, activation='softmax'))

        # model.layers[0].trainable = False

        model.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        model.summary()

        return model


    def meshnet_test(self):
        # base_model = ResNet50(include_top=False, input_shape=(
        #     self.CFG['img_height'], self.CFG['img_width'], 3), pooling='avg', weights=self.CFG['pretrained_weights'])

        # Define model with different applications
        model = Sequential()
        model.add(Input(shape=(150, 150, 3)))
        # model.add(base_model)
        
        # image = Input(shape=(150, 150, 3))
        # x = Conv2D(3, kernel_size=(3,3), padding='same', activation='relu')

        # MeshNet-inspired layers adapted for 2D, including dilation
        # model.add(Dense(512, activation='relu'))

        # model.add(layers.Conv2D(2048,(3,3),padding='same',activation='relu'))
        # model.add(layers.Conv2D(1024,(3,3),padding='same',activation='relu'))
        # model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
        # Layer 1: Convolutional + Relu + BatchNorm + Dropout
        
        model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())

        # Layers 2-6: Repetition of Convolutional + Relu + BatchNorm + Dropout with increasing dilation rates
        """
        Our architecture entirely consists of what [7] used as a context module but we modified it to use 3D dilated convolutions.
        https://arxiv.org/pdf/1612.00940.pdf (MeshNet Section)
        """
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(1, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())
        # Layer 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())
        # Layer 4
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(4, 4)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())
        # Layer 5
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(8, 8)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())
        # Layer 6
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(16, 16)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())

        # dilation_rates = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
        # for rate in dilation_rates:
        #     model.add(Conv2D(filters=64, kernel_size=(3, 3),
        #             padding='same', dilation_rate=rate))
        #     model.add(Activation('relu'))
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.3))

        # Layer 7: Convolutional + BatchNorm + Activation + Dropout
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(layers.MaxPooling2D())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        # model.add(layers.MaxPooling2D())

        # model.add(GlobalMaxPooling2D())

        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        model.add(GlobalMaxPooling2D())
        # model.add(GlobalAveragePooling2D())

        
        

        model.add(Dense(2, activation='softmax'))

        # model.layers[0].trainable = False

        model.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        model.summary()

        return model

    def ensemble(self, snapshot_path):
        import itertools
        import glob
        
        
        
        modelfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*.hdf5', recursive=True)]))
        modelnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, modelfiles))

        models = []
        input_size = []
        yModels = []
        for i, m in enumerate(modelfiles):
            
            model = load_model(m)
            # model._name = f'{pathlib.Path(os.path.basename(m)).stem}_{i}'
            model._name = str(i)
            models.append(model)
            # input_size.append(Input(shape=model.input_shape[1:])) # h*w*c
            # yModels.append(model(input_size[i]))

        model_input = Input(shape=models[0].input_shape[1:]) # h*w*c
        yModels=[model(model_input) for model in models]
        # averaging outputs
        yAvg = average(yModels)
        
        # build model from same input and avg output
        modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

        modelEns.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])
        
        
        if not os.path.exists(f'{snapshot_path}/ensemble'):
                os.makedirs(f'{snapshot_path}/ensemble', exist_ok=True)

        model_path = f'{snapshot_path}/ensemble/Ensemble.hdf5'

        modelEns.save(model_path)
        modelEns=load_model(model_path)
        modelEns.summary()
   


    
    

    
