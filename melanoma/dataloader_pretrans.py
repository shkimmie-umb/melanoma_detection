import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import decode_image
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import melanoma as mel

class DataLoaderFast(torch.utils.data.Dataset):
    def __init__(self, folderpath: str, pre_transform: transforms = None, post_transform: transforms = None) -> None:
        self.data = torchvision.datasets.ImageFolder(folderpath)

        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.X = []
        for X in self.data.imgs:
            self.X.append(mel.Parser.encode(pre_transform(Image.open(X[0]))))
        self.targets = self.data.targets

        

    def __len__(self) -> int:
        self.length = len(self.data)
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """ Returns a tuple of image and label."""
        X, y = self.post_transform(mel.Parser.decode(self.X[idx])), self.targets[idx]

        return X, torch.tensor(y)