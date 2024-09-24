import torch
import torchvision.transforms as transforms
from torchvision.io import decode_image
from torchvision.transforms.functional import pil_to_tensor
import melanoma as mel

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, combined_data: list, mode: str = 'Train', transform: transforms = None) -> None:
        self.combined_data = combined_data
        self.transform = transform
        self.dataset = None
        self.mode = mode

        if mode == 'Train':
            self.length = len(combined_data['trainimages'])
        elif mode == 'Val':
            self.length = len(combined_data['validationimages'])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """ Returns a tuple of image and label."""
        # if self.dataset is None:
        #     self.dataset = h5py.File(self.hdf5_file, mode="r", swmr=True)

        # image = self.dataset["images"][idx]
        # label = self.dataset["labels"][idx]

        if self.mode == 'Train':
            image = mel.Parser.decode(self.combined_data["trainimages"][idx])
            label = self.combined_data["trainlabels"][idx]
        elif self.mode == 'Val':
            image = mel.Parser.decode(self.combined_data["validationimages"][idx])
            label = self.combined_data["validationlabels"][idx]

        if self.transform and self.mode == 'Train':
            image = self.transform['Train'](image)
        if self.transform and self.mode == 'Val':
            image = self.transform['Val'](image)

        return image, label