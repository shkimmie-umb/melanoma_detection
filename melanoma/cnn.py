import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# import Model
from .model import Model as Base_Model
from .augmentationStrategy import *

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Activation, Dropout, BatchNormalization,
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, add
)
from keras.layers.merge import concatenate

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
        
import melanoma as mel

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(Base_Model):

    def __init__(self, CFG, train_images=None, train_labels=None, val_images=None, val_labels=None, test_images=None, test_labels=None):
        super().__init__(CFG, train_images, train_labels, val_images, val_labels, test_images, test_labels)

         

    def augmentation(self, img_height, img_width, rotation, zoom):
         return self._augmentation_strategy.augmentation(img_height, img_width, rotation, zoom)
    
    def CNN_3conv2(self):
        # Our input feature map is 64x64x3: 64x64 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(self.img_height, self.img_width, 3))

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Convolution2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        # Configure and compile the model
        model = Model(img_input, output)

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model


    def CNN_3conv(self, img_height, img_width, class_names, augmentation_type):
            # ### Create the first model
            # #### Creating a CNN model, which can accurately detect 9 classes present in the dataset.
            # Using ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. 


            # CNN Model - Initial
            model=models.Sequential()
            # scaling the pixel values from 0-255 to 0-1
            ##Todo: change img_height, img_width to get from loadTrainData return values (tf.data.Dataset)
            model.add(layers.experimental.preprocessing.Rescaling(scale=1./255,input_shape=(img_height,img_width,3)))

            if augmentation_type is not None:
                # print(augmentation_type)
                model.add(Augmentation(augmentation_type).augmentation(img_height, img_width))
            

            # Convolution layer with 32 features, 3x3 filter and relu activation with 2x2 pooling
            model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D())

            # Convolution layer with 64 features, 3x3 filter and relu activation with 2x2 pooling
            model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D())

            # Convolution layer with 128 features, 3x3 filter and relu activation with 2x2 pooling
            model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D())

            #Dropout layer with 50% Fraction of the input units to drop.
            model.add(layers.Dropout(0.5))

            model.add(layers.Flatten())
            model.add(layers.Dense(256,activation='relu'))

            #Dropout layer with 25% Fraction of the input units to drop.
            model.add(layers.Dropout(0.25))

            model.add(layers.Dense(len(class_names),activation='softmax'))

            return model

    
    def resnet50(self, ResNet50_name):
        data_gen = ImageDataGenerator(rotation_range = self.CFG['ROTATION_RANGE'], zoom_range = self.CFG['ZOOM_RANGE'],
                                      width_shift_range = self.CFG['WSHIFT_RANGE'], height_shift_range = self.CFG['HSHIFT_RANGE'],
                                      horizontal_flip = self.CFG['HFLIP'], vertical_flip = self.CFG['VFLIP'])
        data_gen.fit(self.train_images)
        
        resnet50_out = ResNet50(
            include_top=False,
            # include_top=True,
            input_shape=(self.CFG['img_height'], self.CFG['img_width'], 3),
            # input_shape=(64, 64, 3),
            pooling = 'avg',
            weights='imagenet'
        )

        Resnet50_model = super().build_model(
            base_model = resnet50_out,
            base_model_name = ResNet50_name,
            model_optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        )

        return data_gen, Resnet50_model

    def transformer(self, network):
        # Define model with different applications
        tf.keras.backend.clear_session()
        model = Sequential()
        #vgg-16 , 80% accuracy with 100 epochs
        # model.add(VGG16(input_shape=(224,224,3),pooling='avg',classes=1000,weights=vgg16_weights_path))
        #resnet-50 , 87% accuracy with 100 epochs
        model.add(network(
                include_top=False,
                input_tensor=None,
                input_shape=(self.CFG['img_height'], self.CFG['img_width'], 3),
                pooling='avg',
                # classes=self.CFG['num_classes'],
                weights=self.CFG['pretrained_weights']
        ))
        # model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))

        model.layers[0].trainable = False
        model.summary()

        model.compile(optimizer=self.CFG['model_optimizer'], loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        

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

    def meshnet_kerasstyle(self, include_top=False, weights='imagenet', input_tensor=None, \
        input_shape=None, pooling='max', classes=1000, **kwargs):

        # from tensorflow.keras.applications import get_submodules_from_kwargs
        # from tensorflow.keras.applications import imagenet_utils
        # from tensorflow.keras.applications.imagenet_utils import decode_predictions
        # from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
        
        
        from keras.applications.imagenet_utils import obtain_input_shape
        
        input_shape = obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=tf.keras.backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not tf.keras.backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # Block 1
        x = layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block1_conv1')(img_input)
        x = layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(2, 2),
                        name='block3_conv1')(x)
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(2, 2),
                        name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(2, 2),
                        name='block3_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(4, 4),
                        name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(4, 4),
                        name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(4, 4),
                        name='block4_conv3')(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(8, 8),
                        name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(8, 8),
                        name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(8, 8),
                        name='block5_conv3')(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 6
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(16, 16),
                        name='block6_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(16, 16),
                        name='block6_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(16, 16),
                        name='block6_conv3')(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 7
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block7_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block7_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block7_conv3')(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)
        # x = layers.BatchNormalization()(x)

        # Block 8
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block8_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block8_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        dilation_rate=(1, 1),
                        name='block8_conv3')(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

        x = layers.Dense(512, activation='relu')(x)
        x= layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        # model.add(Dense(1, activation='sigmoid'))
        x = layers.Dense(2, activation='softmax')(x)


        if include_top:
            # Classification block
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(4096, activation='relu', name='fc1')(x)
            x = layers.Dense(4096, activation='relu', name='fc2')(x)
            x = layers.Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = tf.keras.utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = models.Model(inputs, x, name='meshnet')

        WEIGHTS_PATH = '/hpcstor6/scratch01/s/sanghyuk.kim001/weights/'
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'

        # Load weights.
        # if weights == 'imagenet':
        #     if include_top:
        #         weights_path = tf.keras.utils.get_file(
        #             'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        #             WEIGHTS_PATH,
        #             cache_subdir='models',
        #             file_hash='64373286793e3c8b2b4e3219cbf3544b')
        #     else:
        #         weights_path = tf.keras.utils.get_file(
        #             'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #             WEIGHTS_PATH_NO_TOP,
        #             cache_subdir='models',
        #             file_hash='6d6bbae143d832006294945121d1f1fc')
        #     model.load_weights(weights_path)
        #     if tf.keras.backend.backend() == 'theano':
        #         tf.keras.utils.convert_all_kernels_in_model(model)
        # elif weights is not None:
        #     model.load_weights(weights)


        # model.layers[0].trainable = False
        model.summary()

        model.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        return model
    def meshnet_test(self, network=None):
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
        
        model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
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
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', dilation_rate=(1, 1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Layer 3
        model.add(Conv2D(filters=256, kernel_size=(3, 3),
                    padding='same', dilation_rate=(2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Layer 4
        model.add(Conv2D(filters=512, kernel_size=(3, 3),
                    padding='same', dilation_rate=(4, 4)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Layer 5
        model.add(Conv2D(filters=512, kernel_size=(3, 3),
                    padding='same', dilation_rate=(8, 8)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # Layer 6
        model.add(Conv2D(filters=512, kernel_size=(3, 3),
                    padding='same', dilation_rate=(16, 16)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # dilation_rates = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
        # for rate in dilation_rates:
        #     model.add(Conv2D(filters=64, kernel_size=(3, 3),
        #             padding='same', dilation_rate=rate))
        #     model.add(Activation('relu'))
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.3))

        # Layer 7: Convolutional + BatchNorm + Activation + Dropout
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))

        # model.add(GlobalMaxPooling2D())

        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        model.add(GlobalMaxPooling2D())
        

        model.add(Dense(2, activation='softmax'))

        # model.layers[0].trainable = False

        model.compile(optimizer=self.CFG['model_optimizer'],
                    loss=self.CFG['loss'], metrics=self.CFG['metrics'])

        model.summary()

        return model

    # def meshnet_dilation(input_shape, apply_softmax=True, input_tensor=None, classes):

    #     if input_tensor is None:
    #         model_in = Input(shape=input_shape)
    #     else:
    #         if not K.is_keras_tensor(input_tensor):
    #             model_in = Input(tensor=input_tensor, shape=input_shape)
    #         else:
    #             model_in = input_tensor

    #     h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    #     h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    #     h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    #     h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    #     h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    #     h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    #     h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    #     h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    #     h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    #     h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    #     h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    #     h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    #     h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    #     h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    #     h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    #     h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    #     h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    #     h = Dropout(0.5, name='drop6')(h)
    #     h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    #     h = Dropout(0.5, name='drop7')(h)
    #     h = Convolution2D(classes, 1, 1, name='final')(h)
    #     h = ZeroPadding2D(padding=(1, 1))(h)
    #     h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    #     h = ZeroPadding2D(padding=(1, 1))(h)
    #     h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    #     h = ZeroPadding2D(padding=(2, 2))(h)
    #     h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    #     h = ZeroPadding2D(padding=(4, 4))(h)
    #     h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    #     h = ZeroPadding2D(padding=(8, 8))(h)
    #     h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    #     h = ZeroPadding2D(padding=(16, 16))(h)
    #     h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    #     h = ZeroPadding2D(padding=(1, 1))(h)
    #     h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    #     logits = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    #     if apply_softmax:
    #         model_out = softmax(logits)
    #     else:
    #         model_out = logits

    #     model = Model(input=model_in, output=model_out, name='dilation_camvid')

    #     return model


    
    def trainData(self, train_ds, val_ds, img_height, img_width, class_names, augmentation_type, epochs=20):
        super().__init__(train_ds, val_ds, epochs)
        # augmentation_type = simple_augmentation()
        model = self.CNN_3conv(img_height, img_width, class_names, augmentation_type)
        model.compile(optimizer='adam',
		              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		              metrics=['accuracy'])
        model.summary()

        

		# Training the model
		# epochs = 20
        history = model.fit(
		  train_ds,
		  validation_data=val_ds,
		  epochs=epochs
		)

        return history
    
    def train(self, data_gen_X_train, data_gen_X_val, X_train, y_train, X_val, y_val):
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        batch_size = 64
        epochs = 30
        model = self.CNN_3conv2()
        history = model.fit(data_gen_X_train.flow(X_train,y_train, batch_size=batch_size),
                                    epochs = epochs, validation_data = data_gen_X_val.flow(X_val, y_val),
                                    verbose = 0, steps_per_epoch=(X_train.shape[0] // batch_size),
                                    callbacks=[mel.SilentTrainingCallback, learning_rate_reduction])
        return model, history
    
