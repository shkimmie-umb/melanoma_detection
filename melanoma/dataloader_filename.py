import torch
import torchvision
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import pathlib

class ImageFolder_filename(ImageFolder):


    def __getitem__(self, index):
            # Get the original tuple (image and label)
            original_tuple = super(ImageFolder_filename, self).__getitem__(index)
            
            # Access the image file path (self.samples contains (path, label))
            file_path = self.samples[index][0]
            
            # Return image, label, and file path
            return original_tuple + (file_path,)