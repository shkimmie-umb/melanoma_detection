from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

class AugmentationStrategy:
    def augmentation(self, img_height, img_width, rotation, zoom):
        pass

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
class no_augmentation(AugmentationStrategy):
    def augmentation(self):

        print("Will not use data augmentation")
class Augmentation:
    def __init__(self, strategy):
        self._strategy = strategy

    def augmentation(self, img_height, img_width, rotation=0.1, zoom=0.1):
        return self._strategy.augmentation(img_height, img_width, rotation, zoom)



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