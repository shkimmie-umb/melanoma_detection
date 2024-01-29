from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import Augmentor as am
import albumentations as A
import pathlib
import os

class AugmentationStrategy:
    # def augmentation(self, img_height, img_width, rotation, zoom):
    #     pass
    def augmentation(self, input_img=None, crop_height=None, crop_width=None, zoomout=None, zoomin=None, p_scaling=None, p_rotation=None, p_hflip=None, p_vflip=None, p_randomBrightnessContrast=None, resize_ratio=None):
        pass
    def augment_and_save(self):
        pass

class crop_flip_brightnesscontrast(AugmentationStrategy):
    @staticmethod
    def augmentation(input_img, crop_height, crop_width, zoomout, zoomin, p_scaling, p_rotation, p_hflip, p_vflip,
    p_randomBrightnessContrast, resize_ratio=None):
        transform = A.Compose([
            A.Rotate(limit=(-90, 90), p=p_rotation),
            A.Affine(scale=(zoomout, zoomin), p=p_scaling),
            A.RandomCrop(width=crop_width, height=crop_height),
            A.VerticalFlip(p=p_vflip),
            A.HorizontalFlip(p=p_hflip),
            A.RandomBrightnessContrast(p=p_randomBrightnessContrast),
        ])

        transformed = transform(image=input_img)

        return transformed

class crop_flip(AugmentationStrategy):
    @staticmethod
    def augmentation(input_img, crop_height, crop_width, zoomout, zoomin, p_scaling, p_rotation, p_hflip, p_vflip, p_randomBrightnessContrast=None, resize_ratio=None):
        img_width = input_img.shape[1]
        img_height = input_img.shape[0]

        transform = A.Compose([
            A.Resize(int(img_height * resize_ratio), int(img_width * resize_ratio)),
            A.Rotate(limit=(-45, 45), p=p_rotation),
            A.Affine(scale=(zoomout, zoomin), p=p_scaling),
            A.RandomCrop(width=crop_width, height=crop_height),
            A.VerticalFlip(p=p_vflip),
            A.HorizontalFlip(p=p_hflip),
        ])

        transformed = transform(image=input_img)

        return transformed

class simple_augmentation(AugmentationStrategy):
    @staticmethod
    def augmentation(img_height, img_width, rotation=0.1, zoom=0.1):
        data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
            layers.experimental.preprocessing.RandomRotation(rotation),
            layers.experimental.preprocessing.RandomZoom(zoom),
        ]
        )
        return data_augmentation

class generate_augmentation(AugmentationStrategy):
    @staticmethod
    def augment_and_save(class_names, path, numSamplesToAdd, probability=0.7, max_left_rotation=10, max_right_rotation=10):
        
        # Set path of database in which train and test folders are located
        trainPath = path + '/Train'
        print(trainPath)

        for idx, name in enumerate(class_names):
            print('idx: ', idx, 'name: ', name)
            if (os.path.exists(os.path.join(trainPath + '/' + class_names[idx] + '/' + 'output'))) is False:
                pipeln = am.Pipeline(trainPath + '/' + name)
                pipeln.rotate(probability, max_left_rotation, max_right_rotation)
                
                pipeln.sample(numSamplesToAdd)  #Adding # samples per class to make sure that none of the classes are sparse
                print('Augmented', numSamplesToAdd, ' images in each class')
            else:
                print('Augmentation for ' + class_names[idx] + ' already exists')

                

        
        data_dir_train = pathlib.Path(trainPath + '/')
        image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
        print("Newly generated images with the Augmentor library:", image_count_train)


class no_augmentation(AugmentationStrategy):
    def augmentation(self):

        print("Will not use data augmentation")
class Augmentation:
    def __init__(self, strategy):
        self._strategy = strategy

    def augmentation(self, input_img=None, crop_height=None, crop_width=None, zoomout=None, zoomin=None, p_scaling=None, p_rotation=None, p_hflip=None, p_vflip=None, p_randomBrightnessContrast=None, resize_ratio=None):
        return self._strategy.augmentation(input_img, crop_height, crop_width, zoomout, zoomin, p_scaling, p_rotation, p_hflip, p_vflip, p_randomBrightnessContrast, resize_ratio)
    
    def augment_and_save(self, class_names, path, numSamplesToAdd, probability=0.7, max_left_rotation=10, max_right_rotation=10):
        return self._strategy.augment_and_save(class_names, path, numSamplesToAdd, probability, max_left_rotation, max_right_rotation)
    




#     strategy: Strategy  ## the strategy interface

#     def setAugmentation(self, strategy: Strategy = None) -> None:
#         if strategy is not None:
#             self.strategy = strategy
#         else:
#             self.strategy = Default()

#     def executeAugmentation(self) -> str:
#         print(self.strategy.execute())

# ## Strategy interface
# class Strategy(ABC):
#     @abstractmethod
#     def execute(self) -> str:
#         pass

    # def __init__(self):
    #     pass

    # def basic_augmentation(self):
    #     data_augmentation = keras.Sequential(
    #     [
    #         layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    #         layers.experimental.preprocessing.RandomRotation(0.1),
    #         layers.experimental.preprocessing.RandomZoom(0.1),
    #     ]
    #     )
    #     return data_augmentation