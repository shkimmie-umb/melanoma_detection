# import Model
from .model import Model as Base_Model
import melanoma as mel
import torch.nn as nn

import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights



class CNN(Base_Model):

    def __init__(self):
        super().__init__()

    @staticmethod
    def model_caller(classifier):
        classifierDict = {
            mel.NetworkType.ResNet50.name: resnet50,
            mel.NetworkType.VGG19.name: vgg19,
            mel.NetworkType.WideResNet50_2.name: wide_resnet50_2,
        }
        model = classifierDict[classifier]

        return model

    @staticmethod
    def transfer(network, weights, CFG):
        
        model_ft = network(weights=weights)

        for param in model_ft.parameters():
            param.requires_grad = False

        # Freeze only the convolutional layers of the pre-trained model
        # for param in model_ft.parameters():
        #     if isinstance(param, nn.Conv2d):
        #         param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        model_ft.fc = nn.Linear(num_ftrs, CFG['num_classes'])
        # model_ft.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 256),  # Additional linear layer with 256 output features
        #     nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
        #     nn.Dropout(0.5),               # Dropout layer with 50% probability
        #     nn.Linear(256, num_classes)    # Final prediction fc layer
        # )

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
   


    
    

    
