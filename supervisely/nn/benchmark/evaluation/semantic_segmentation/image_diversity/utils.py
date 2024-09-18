from PIL import Image
import warnings

try:
    from torch.utils.data import Dataset
except ImportError:
    warnings.warn(
        "Pytorch is not installed (ignore this warning if you are not going to use semantic segmentation model benchmark)"
    )


class ImgDataset(Dataset):
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
